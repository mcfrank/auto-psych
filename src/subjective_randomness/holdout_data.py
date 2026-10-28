"""Response generation and data preparation for holdout recovery.

Helpers that generate ground-truth responses from seed models with fixed
parameters, strip identity columns, and validate raw-column contracts. What
is task-specific (the columns, the prediction's name, how a participant is
simulated) comes from the project's ``TaskSpec`` (``src.tasks``).
These are the data-preparation building blocks consumed by the experiment
orchestrator and trajectory evaluator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.models.model_manifest import read_manifest_names
from src.models.data_binding import make_stim_data
from src.models.model_loading import load_pymc_model
from src.tasks import TaskSpec

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


def resolve_generating_params(
    spec: Any, seed_params: Mapping[str, Mapping[str, float]]
) -> Dict[str, Dict[str, float]]:
    """Turn a config's ``gt_models`` spec into per-model fixed params: null ->
    every model of ``seed_params`` (the config's); a list of names, or a
    name -> params mapping whose null params, -> those models' ``seed_params``.
    A model with no parameters anywhere raises (KeyError)."""
    if spec is None:
        return {name: dict(params) for name, params in seed_params.items()}
    if isinstance(spec, (list, tuple)):
        return {name: dict(seed_params[name]) for name in spec}
    if isinstance(spec, Mapping):
        return {
            name: dict(params) if params else dict(seed_params[name])
            for name, params in spec.items()
        }
    raise TypeError(
        f"generating_models must be null, a list of names, or a name->params "
        f"mapping; got {type(spec).__name__}."
    )


def _raw_eval_rows(
    stimuli: Sequence[Mapping[str, Any]], task: TaskSpec
) -> List[Dict[str, Any]]:
    """Build raw stimulus rows for evaluation (no featurization).

    Each model computes its own features through its ``compute_features`` or
    ``prepare_observed`` hook when ``make_stim_data`` is called.
    """
    if not stimuli:
        raise ValueError("No stimuli provided.")
    return task.stimulus_rows(stimuli)


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
    stimuli: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    *,
    task: TaskSpec,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic prediction (the task's ``prediction_var``, ``p_left`` for
    the coin-flip task) per stimulus for a seed model with fixed params.

    Uses raw stimulus rows; the model computes its own features through its
    ``compute_features`` or ``prepare_observed`` hook.
    """
    import pymc as pm

    model = load_pymc_model(model_name, models_dir)
    _require_exact_params(model, params)
    rows = _raw_eval_rows(stimuli, task)
    stim_data = make_stim_data(model, rows)

    with model:
        pm.set_data(stim_data)
    fixed = pm.do(model, {name: np.asarray(value) for name, value in params.items()})
    with fixed:
        prior = pm.sample_prior_predictive(
            draws=1, var_names=[task.prediction_var], random_seed=seed
        )
    return np.asarray(prior.prior[task.prediction_var].values).reshape(-1)


def generate_responses(
    model_name: str,
    models_dir: Path,
    stimuli: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    n_participants: int,
    *,
    task: TaskSpec,
    seed: int = 0,
    participant_id_offset: int,
) -> List[Dict[str, Any]]:
    """Generate synthetic responses from a seed model with fixed parameters.

    The task simulates its participants (``TaskSpec.simulate``): for the
    coin-flip task each sees each pair in a random left/right order, for the
    number game each answers each (set, target) cell. Participant ids run from
    ``participant_id_offset``, so that ids stay unique when several
    experiments' responses are pooled.

    Returns rows with only the task's raw columns, plus generating_model.
    """
    if n_participants < 1:
        raise ValueError(f"n_participants must be >= 1, got {n_participants}.")

    def predict(rows: List[Dict[str, Any]]) -> np.ndarray:
        return p_left_fixed_params(model_name, models_dir, rows, params, task=task, seed=seed)

    rows = task.simulate(
        predict, list(stimuli), n_participants, np.random.default_rng(seed), participant_id_offset
    )
    return [{**row, GENERATING_MODEL_COLUMN: model_name} for row in rows]


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
