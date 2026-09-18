"""Response generation and data preparation for holdout recovery.

Helpers that generate ground-truth responses from seed models with fixed
parameters, strip identity columns, and validate raw-column contracts.
These are the data-preparation building blocks consumed by the experiment
orchestrator and trajectory evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.models.model_manifest import read_manifest_names
from src.models.pymc_inference import load_pymc_model, make_stim_data
from src.pipelines.outer_loop.columns import RAW_RESPONSE_COLUMNS

PROJECT_ID = "subjective_randomness"

# ``generate_responses`` tags every row with the name of the model that produced
# it. That tag is the held-out model's identity: it must never reach the agents'
# tree (data/responses.csv and the pooled model_loop/responses.csv derived from
# it are listed column-by-column in every candidate's and critic's context).
GENERATING_MODEL_COLUMN = "generating_model"


# ─────────────────────────────────────────────
# Seed-model helpers (local to avoid a dependency on the research library)
# ─────────────────────────────────────────────


def seed_exclusion(gt_model: str, pool_dir: Path) -> Tuple[str, ...]:
    """Which seed models to withhold from experiment 1, by manifest name.

    ``pool_dir`` must be the SAME directory seeding reads. The array scrubs the
    held-out model out of the pool manifest it was told about, so a membership
    test against a different pool would still say "exclude it" while seeding
    reads a manifest that no longer lists it — and the exclusion raises
    ``exclude names models not in the seed manifest``. A ground truth absent
    from the pool (already scrubbed, superseded by a consolidation, or an
    impossible theory) needs nothing withheld.
    """
    return (gt_model,) if gt_model in seed_model_names(pool_dir) else ()


def seed_model_names(seed_models_dir: Path) -> List[str]:
    """Read the ordered seed-model names from the directory's manifest."""
    return read_manifest_names(seed_models_dir)


def _default_params_from_file(path: Path) -> Dict[str, float]:
    """Parse a family's ``DEFAULT_PARAMS`` from source WITHOUT importing it."""
    import ast as _ast

    tree = _ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in _ast.walk(tree):
        target = None
        if isinstance(node, _ast.AnnAssign):
            target = getattr(node.target, "id", None)
        elif isinstance(node, _ast.Assign):
            target = next(
                (t.id for t in node.targets if isinstance(t, _ast.Name)), None
            )
        if target == "DEFAULT_PARAMS":
            return dict(_ast.literal_eval(node.value))
    raise ValueError(f"No DEFAULT_PARAMS literal found in {path}")


def _family_default_params(
    name: str, gt_family_dir: Optional[Path] = None
) -> Dict[str, float]:
    """``DEFAULT_PARAMS`` of the pure-Python model family named ``name``."""
    import importlib

    if gt_family_dir is not None:
        pristine = Path(gt_family_dir) / f"{name}.py"
        if pristine.exists():
            return _default_params_from_file(pristine)
    module = importlib.import_module(f"src.subjective_randomness.model_families.{name}")
    return dict(module.DEFAULT_PARAMS)


def resolve_generating_params(
    spec: Any,
    seed_models_dir: Path,
    gt_family_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Turn a config's ``generating_models`` spec into per-model fixed params."""
    if spec is None:
        return {
            name: _family_default_params(name, gt_family_dir)
            for name in seed_model_names(seed_models_dir)
        }
    if isinstance(spec, (list, tuple)):
        return {name: _family_default_params(name, gt_family_dir) for name in spec}
    if isinstance(spec, Mapping):
        return {
            name: (
                dict(params) if params else _family_default_params(name, gt_family_dir)
            )
            for name, params in spec.items()
        }
    raise TypeError(
        f"generating_models must be null, a list of names, or a name->params "
        f"mapping; got {type(spec).__name__}."
    )


def _raw_eval_rows(
    stimuli: Sequence[Mapping[str, str]],
) -> List[Dict[str, Any]]:
    """Build raw stimulus rows for evaluation (no featurization).

    Each model computes its own features through its ``compute_features`` or
    ``prepare_observed`` hook when ``make_stim_data`` is called.
    """
    if not stimuli:
        raise ValueError("No stimuli provided.")
    return [
        {
            "sequence_a": stim["sequence_a"],
            "sequence_b": stim["sequence_b"],
            "chose_left": 0,
        }
        for stim in stimuli
    ]


def _require_exact_params(model: Any, params: Mapping[str, float]) -> None:
    """Fail loudly unless ``params`` names exactly the model's free parameters."""
    free = {rv.name for rv in model.free_RVs}
    given = set(params)
    if given != free:
        missing = sorted(free - given)
        extra = sorted(given - free)
        raise ValueError(
            f"Generating params must name exactly the model's free parameters "
            f"{sorted(free)}. Missing: {missing}. Unexpected: {extra}."
        )


def p_left_fixed_params(
    model_name: str,
    models_dir: Path,
    stimuli: Sequence[Mapping[str, str]],
    params: Mapping[str, float],
    *,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic ``p_left`` per stimulus for a seed model with fixed params.

    Uses raw stimulus rows; the model computes its own features through its
    ``compute_features`` or ``prepare_observed`` hook.
    """
    import pymc as pm

    model = load_pymc_model(model_name, models_dir)
    _require_exact_params(model, params)
    rows = _raw_eval_rows(stimuli)
    stim_data = make_stim_data(model, rows)

    with model:
        pm.set_data(stim_data)
    fixed = pm.do(model, dict(params))
    with fixed:
        prior = pm.sample_prior_predictive(
            draws=1, var_names=["p_left"], random_seed=seed
        )
    return np.asarray(prior.prior["p_left"].values).reshape(-1)


def generate_responses(
    model_name: str,
    models_dir: Path,
    stimuli: Sequence[Mapping[str, str]],
    params: Mapping[str, float],
    n_participants: int,
    *,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Generate synthetic responses from a seed model with fixed parameters.

    Returns rows with only raw columns (sequence_a, sequence_b, participant_id,
    trial_index, chose_left) plus generating_model.
    """
    if n_participants < 1:
        raise ValueError(f"n_participants must be >= 1, got {n_participants}.")

    p_left = p_left_fixed_params(model_name, models_dir, stimuli, params, seed=seed)
    rng = np.random.default_rng(seed)

    rows: List[Dict[str, Any]] = []
    for participant in range(n_participants):
        draws = rng.random(len(stimuli)) < p_left
        for trial_index, (stim, chose_left) in enumerate(
            zip(stimuli, draws)
        ):
            rows.append(
                {
                    "sequence_a": stim["sequence_a"],
                    "sequence_b": stim["sequence_b"],
                    "participant_id": participant,
                    "trial_index": trial_index,
                    "chose_left": int(chose_left),
                    "generating_model": model_name,
                }
            )
    return rows


def strip_generating_model(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Copy ``rows`` without the ``generating_model`` column.

    The input rows are not modified — the non-holdout recovery harness keeps
    reading the tag from its own rows. Rows that never carried the column pass
    through unchanged.
    """
    return [
        {key: value for key, value in row.items() if key != GENERATING_MODEL_COLUMN}
        for row in rows
    ]


def strip_to_raw_columns(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Copy ``rows`` keeping only :data:`RAW_RESPONSE_COLUMNS`.

    Fails loudly if a row lacks one of them: a raw CSV missing a sequence
    column would leave every model unable to compute anything, and the useful
    place to find that out is here.
    """
    rows = list(rows)
    if rows:
        missing = [c for c in RAW_RESPONSE_COLUMNS if c not in rows[0]]
        if missing:
            raise ValueError(
                f"generated rows lack raw columns {missing}; "
                f"got {sorted(rows[0])}"
            )
    return [{c: row[c] for c in RAW_RESPONSE_COLUMNS} for row in rows]


def validate_raw_pool_models(pool_dir: Path) -> None:
    """Raise if any model in ``pool_dir`` cannot bind a raw stimulus row.

    Every model must compute its own features via ``compute_features`` or
    ``prepare_observed``. A model that expects columns the raw CSV does not
    carry would fail at fit time deep inside a sweep; catching it here fails
    at config resolution with the model's name and the missing columns.
    """
    from src.models.pymc_inference import (
        MissingStimulusColumns,
        load_pymc_model,
        make_stim_data,
    )

    raw_row = {c: "0" for c in RAW_RESPONSE_COLUMNS}
    raw_row["sequence_a"] = "HHT"
    raw_row["sequence_b"] = "THT"
    for name in read_manifest_names(pool_dir):
        try:
            model = load_pymc_model(name, pool_dir)
        except Exception as exc:
            raise ValueError(
                f"raw pool model {name!r} in {pool_dir} failed to load: {exc}"
            ) from exc
        try:
            make_stim_data(model, [raw_row])
        except MissingStimulusColumns as exc:
            raise ValueError(
                f"raw pool model {name!r} cannot bind a raw row — missing "
                f"columns: {list(exc.missing)}. Every model must compute its "
                f"own features via compute_features or prepare_observed."
            ) from exc


def _require_no_generating_model_column(responses_path: Path) -> None:
    """Fail loudly if an agent-facing responses CSV names its generator."""
    with Path(responses_path).open(encoding="utf-8", newline="") as f:
        header = [column.strip() for column in f.readline().strip().split(",")]
    if GENERATING_MODEL_COLUMN in header:
        raise RuntimeError(
            f"{responses_path} carries a {GENERATING_MODEL_COLUMN!r} column, which "
            f"names the held-out model to every agent that opens the file. The "
            f"holdout harness must write agent-facing responses without it."
        )
