"""Grouped cross-validation: the inner loop's alternative model-selection criterion.

Trial-level PSIS-LOO asks how well a model predicts another answer to a
stimulus it was fitted to: every other answer to that stimulus, and for a
model with participant parameters the rest of that participant's session, is
in the fit, so it is close to in-sample fit with a complexity penalty. Grouped
CV holds out whole units: the training trials are split into ``cv_folds``
folds of whole units, each model is fitted on the other folds, and each
held-out trial is scored by its log posterior predictive density. The sum over
trials is the model's ELPD-CV. (auto-rsa found the trial-level LOO winner
59-74 nats worse on held-out conditions than the models that generalised best,
and switched; PI decision 2026-10-08 there.)

``cv_group`` names the unit: ``"stimulus"`` (the task's stimulus identity,
``TaskSpec.stimulus_key``: the unordered pair for the coin-flip task, so both
presentations of a pair fall in one fold; the (set, target) cell for the
number game), or a tuple of CSV columns (``("number_set",)`` holds out whole
example sets, ``("participant_id",)`` whole participants).

The folds depend only on the responses and the settings, so each fold's
training file is content-addressed and its fits are cached like any other.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, List, Literal, Mapping, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Selection:
    """How the inner loop compares models: ``loo`` (trial-level PSIS-LOO, the
    default) or ``grouped_cv`` over ``cv_folds`` folds of ``cv_group`` units."""

    criterion: Literal["loo", "grouped_cv"] = "loo"
    cv_group: Union[str, Tuple[str, ...]] = "stimulus"
    cv_folds: int = 5
    cv_seed: int = 0

    def __post_init__(self) -> None:
        if self.criterion not in ("loo", "grouped_cv"):
            raise ValueError(f"selection criterion must be 'loo' or 'grouped_cv', not {self.criterion!r}")
        if isinstance(self.cv_group, str) and self.cv_group != "stimulus":
            raise ValueError(f"cv_group must be 'stimulus' or a list of columns, not {self.cv_group!r}")
        if self.cv_folds < 2:
            raise ValueError(f"grouped CV needs at least 2 folds, not {self.cv_folds}")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "Selection":
        """From a config block ``{criterion, cv_group, cv_folds, cv_seed}``."""
        group = cfg.get("cv_group", "stimulus")
        return cls(
            criterion=cfg.get("criterion", "loo"),
            cv_group=group if isinstance(group, str) else tuple(group),
            cv_folds=int(cfg.get("cv_folds", 5)),
            cv_seed=int(cfg.get("cv_seed", 0)),
        )

    def as_record(self) -> Dict[str, Any]:
        group = self.cv_group if isinstance(self.cv_group, str) else list(self.cv_group)
        return {"criterion": self.criterion, "cv_group": group, "cv_folds": self.cv_folds, "cv_seed": self.cv_seed}


LOO = Selection()

# Held-out log densities by (fold fit, fold training file): a fold's are
# computed once per process, not at every comparison of the loop.
_HELDOUT_LPD: Dict[Tuple[str, str], np.ndarray] = {}


def cv_units(rows: Sequence[Mapping[str, Any]], cv_group: Union[str, Tuple[str, ...]]) -> List[Hashable]:
    """The unit each row belongs to."""
    if cv_group == "stimulus":
        from src.tasks import task_present_in

        key = task_present_in(rows[0], "stimulus_columns").stimulus_key
        return [key(row) for row in rows]
    missing = [c for c in cv_group if c not in rows[0]]
    if missing:
        raise KeyError(f"cv_group columns {missing} are not in the responses ({sorted(rows[0])})")
    return [tuple(str(row[c]) for c in cv_group) for row in rows]


def assign_folds(units: Sequence[Hashable], k: int, seed: int = 0) -> np.ndarray:
    """The fold of every row: whole units, balanced in rows (largest unit
    first into the lightest fold, ties in a seeded order)."""
    sizes: Dict[Hashable, int] = {}
    for unit in units:
        sizes[unit] = sizes.get(unit, 0) + 1
    if len(sizes) < k:
        raise ValueError(f"grouped CV over {k} folds needs at least {k} units; the responses have fewer units ({len(sizes)})")
    order = list(sizes)
    np.random.default_rng(seed).shuffle(order)
    order.sort(key=lambda unit: -sizes[unit])  # stable: the shuffle breaks ties
    load = np.zeros(k, dtype=int)
    fold_of: Dict[Hashable, int] = {}
    for unit in order:
        f = int(np.argmin(load))
        fold_of[unit] = f
        load[f] += sizes[unit]
    return np.array([fold_of[unit] for unit in units], dtype=int)


def _read_rows(responses_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with Path(responses_path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def fold_training_files(responses_path: Path, selection: Selection) -> Tuple[List[Dict[str, str]], np.ndarray, List[Path]]:
    """The rows, their folds, and one training file per fold (every other fold's
    rows), written once beside the responses under a content-addressed name."""
    header, rows = _read_rows(responses_path)
    folds = assign_folds(cv_units(rows, selection.cv_group), selection.cv_folds, selection.cv_seed)
    digest = hashlib.sha256(
        Path(responses_path).read_bytes() + repr(sorted(selection.as_record().items())).encode()
    ).hexdigest()[:16]
    fold_dir = Path(responses_path).parent / ".cv_folds" / digest
    paths = [fold_dir / f"train_fold{f}.csv" for f in range(selection.cv_folds)]
    if not all(p.exists() for p in paths):
        fold_dir.mkdir(parents=True, exist_ok=True)
        for f, path in enumerate(paths):
            partial = path.with_name(f".{path.name}.partial")
            with partial.open("w", encoding="utf-8", newline="") as out:
                writer = csv.DictWriter(out, fieldnames=header)
                writer.writeheader()
                writer.writerows(row for row, fold in zip(rows, folds) if fold != f)
            partial.replace(path)
    return rows, folds, paths


def cv_pointwise(
    model_names: Sequence[str],
    models_dir: Path,
    responses_path: Path,
    selection: Selection,
    *,
    cache_dir: Any = None,
    **fit_kwargs: Any,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, str]]]:
    """Each model's out-of-fold log posterior predictive density per trial, in
    the responses' row order, and the rows."""
    from src.models.data_binding import make_stim_data
    from src.models.pymc_inference import fit_models_cached

    rows, folds, paths = fold_training_files(responses_path, selection)
    lpd = {name: np.full(len(rows), np.nan) for name in model_names}
    for f, train_path in enumerate(paths):
        held = np.flatnonzero(folds == f)
        fits = fit_models_cached(
            list(model_names), models_dir=models_dir, responses_path=train_path, cache_dir=cache_dir, **fit_kwargs
        )
        held_rows = [rows[i] for i in held]
        for name in model_names:
            fitted = fits[name]
            key = (fitted.fingerprint, str(train_path))
            if key not in _HELDOUT_LPD:
                _HELDOUT_LPD[key] = fitted.heldout_lpd(make_stim_data(fitted.model, held_rows))
            lpd[name][held] = _HELDOUT_LPD[key]
    return lpd, rows
