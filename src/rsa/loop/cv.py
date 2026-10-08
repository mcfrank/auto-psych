"""Grouped cross-validation over the loop's training conditions: the criterion
the loop selects, prunes and exports on (PI decision 2026-10-08).

Sherlock run 1 selected on trial-level PSIS-LOO, which asks how well a model
predicts another trial of a condition it was fitted to. The question that
matters is how well it predicts a condition it has not seen, which is how the
test set is held out (src.rsa.split). In real_rep1 the in-sample winner beat
the six models that generalise best by 177-276 nats and was 59-74 lpd worse
than each of them on held-out conditions.

So the training trials are split into ``k`` folds of whole held-out units
(`src.rsa.split.unit_keys`: a condition of a source's experiment, or a display
of one in a multi-trial experiment), balanced in trials within each source so
every fold holds some of every source. A model is fitted on the other folds
(with the loop's fit rule, `src.rsa.loop.fitting.loop_fit`) and scored on
each fold's trials by the log posterior predictive density
(`src.rsa.evaluate_heldout.heldout_lpd`). Every training trial gets one
out-of-fold lpd; their sum is the model's ELPD-CV, and differences between
models get an SE clustered by unit.

The folds depend only on the training data and the seed, so each fold's fit
is cached like any other and a model's CV is computed once per run.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.rsa.dataset import load_forced_choice
from src.rsa.fit import FitSettings
from src.rsa.split import unit_keys

DEFAULT_FOLDS = 5


def assign_folds(frame: pd.DataFrame, k: int, seed: int = 0) -> np.ndarray:
    """Fold of every row: whole units, balanced in rows within each source."""
    if k < 2:
        raise ValueError(f"grouped CV needs at least 2 folds, not {k}")
    units = unit_keys(frame)
    source = frame["source"] if "source" in frame.columns else pd.Series("pragmods", index=frame.index)
    rng = np.random.default_rng(seed)
    fold_of_unit: Dict[str, int] = {}
    for src in sorted(source.unique()):
        sizes = units[source == src].value_counts()
        if len(sizes) < k:
            raise ValueError(f"source {src} has {len(sizes)} training units, fewer than the {k} folds")
        names = np.array(sorted(sizes.index))
        rng.shuffle(names)
        load = np.zeros(k)
        # Largest first into the lightest fold: balanced rows, whole units.
        for name in sorted(names, key=lambda n: -sizes[n]):
            f = int(np.argmin(load))
            fold_of_unit[name] = f
            load[f] += sizes[name]
    return units.map(fold_of_unit).to_numpy(dtype=int)


@dataclass(frozen=True)
class Folds:
    k: int
    seed: int
    fold: np.ndarray  # (T,) fold of every training trial, in load_forced_choice order
    units: np.ndarray  # (T,) unit id of every training trial (the SE's clusters)
    train_paths: List[Path]  # fold f's training file: the trials of every other fold

    def test_rows(self, f: int) -> np.ndarray:
        return np.where(self.fold == f)[0]


def make_folds(responses_path: Path, out_dir: Path, k: int = DEFAULT_FOLDS, seed: int = 0) -> Folds:
    """Assign the training trials to folds and write each fold's training file
    (once: an existing assignment of the same data, k and seed is reused)."""
    trials = load_forced_choice(responses_path)
    frame = trials.frame
    fold = assign_folds(frame, k, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(k=k, seed=seed, n_trials=len(frame), fold_sizes=np.bincount(fold, minlength=k).tolist(),
                  responses=str(responses_path))
    meta = out_dir / "folds.json"
    if meta.exists() and json.loads(meta.read_text()) != record:
        raise ValueError(f"{meta} records other folds ({json.loads(meta.read_text())}); remove {out_dir} to redo them")
    paths = []
    for f in range(k):
        path = out_dir / f"fold_{f}_train.csv"
        if not path.exists():
            tmp = path.with_suffix(".tmp.csv")
            frame[fold != f].to_csv(tmp, index=False)
            tmp.replace(path)
        paths.append(path)
    meta.write_text(json.dumps(record, indent=1))
    return Folds(k=k, seed=seed, fold=fold, units=pd.factorize(unit_keys(frame))[0], train_paths=paths)


@dataclass
class CVResult:
    pointwise: np.ndarray  # (T,) out-of-fold lpd of every training trial
    converged: bool  # every fold's fit passed the convergence gate (after its refit)

    @property
    def elpd(self) -> float:
        return float(self.pointwise.sum())


def cv_pointwise(model_path: Path, name: str, responses_path: Path, folds: Folds, settings: FitSettings,
                 cache_dir: Path, *, time_limit_sec: Optional[float] = None, workers: int = 1) -> CVResult:
    """Out-of-fold lpd of every training trial. A model failure on any fold
    raises `ModelFailure` (from the fit); fits run ``workers`` at a time."""
    from src.rsa.evaluate_heldout import heldout_lpd
    from src.rsa.loop.fitting import loop_fit
    from src.rsa.model_file import RSAModel

    trials = load_forced_choice(responses_path)
    if len(trials.contexts) != len(folds.fold):
        raise ValueError(f"{responses_path} has {len(trials.contexts)} trials; the folds were made for {len(folds.fold)}")

    def fit_fold(f: int):
        return loop_fit(model_path, name, folds.train_paths[f], settings, cache_dir, time_limit_sec=time_limit_sec)

    if time_limit_sec is None:
        workers = 1  # in-process fits: numpyro's handler stack is not thread-safe
    with ThreadPoolExecutor(max_workers=max(1, min(workers, folds.k))) as pool:
        fits = list(pool.map(fit_fold, range(folds.k)))
    model = RSAModel(model_path, name=name)
    out = np.empty(len(folds.fold))
    for f, fitted in enumerate(fits):
        rows = folds.test_rows(f)
        out[rows] = heldout_lpd(model, fitted, [trials.contexts[i] for i in rows], [trials.choices[i] for i in rows])
    return CVResult(pointwise=out, converged=all(fit.converged for fit in fits))


def compare_cv(results: Dict[str, CVResult], units: np.ndarray) -> Dict[str, dict]:
    """Each model's ELPD-CV, its difference from the best and the unit-clustered SE."""
    from src.models.clustered_se import cluster_dse

    best = max(results, key=lambda n: results[n].elpd)
    out = {}
    for name, r in results.items():
        diff = results[best].elpd - r.elpd
        dse = 0.0 if name == best else float(cluster_dse(results[best].pointwise, r.pointwise, units))
        out[name] = dict(elpd_cv=r.elpd, cv_diff=float(diff), cv_dse=dse, cv_converged=r.converged)
    return out


def ranked(names: Sequence[str], cv: Dict[str, dict]) -> List[str]:
    return sorted(names, key=lambda n: -cv[n]["elpd_cv"])
