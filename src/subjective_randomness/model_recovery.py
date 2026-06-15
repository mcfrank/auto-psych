"""Closed-ended model recovery for the subjective-randomness domain.

"Closed-ended" recovery asks: if a known seed model generated the data, does
the inner model loop — fitting and comparing the *closed* set of seed models,
with no agent-proposed candidates — put its posterior mass back on the true
model?

This module has two halves:

1. Synthetic-data generation. Given a seed model and a fixed parameter vector,
   `generate_responses` featurizes the stimuli, fixes the model's parameters via
   `pm.do`, reads off the deterministic ``p_left`` per stimulus, and samples
   ``chose_left ~ Bernoulli(p_left)`` for each participant. The rows carry the
   *full* feature set, so any seed model can later be fit on them.

2. Recovery orchestration (`run_closed_ended_recovery`, added alongside): loop
   over each generating model, generate data, run the inner loop on the closed
   seed set, and assemble a confusion matrix of recovered posteriors.
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml

from src.models.pymc_inference import load_pymc_model, make_stim_data
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from src.subjective_randomness.config import resolve_path
from src.subjective_randomness.features import featurize_stimulus
from src.subjective_randomness.simulate import load_stimuli

# Columns carried through from each stimulus into the response rows, on top of
# the derived feature columns.
PASSTHROUGH_COLS = ["sequence_a", "sequence_b"]


def seed_model_names(seed_models_dir: Path) -> List[str]:
    """Read the ordered seed-model names from the directory's manifest."""
    manifest_path = Path(seed_models_dir) / "models_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return [entry["name"] for entry in manifest["models"]]


def _family_default_params(name: str) -> Dict[str, float]:
    """``DEFAULT_PARAMS`` of the pure-Python model family named ``name``."""
    module = importlib.import_module(f"src.subjective_randomness.model_families.{name}")
    return dict(module.DEFAULT_PARAMS)


def default_generating_params(
    seed_models_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Default fixed parameters for each seed model.

    Pulled from the matching pure-Python model family's ``DEFAULT_PARAMS``
    (``src.subjective_randomness.model_families.<name>``); its parameter names
    match the PyMC seed model's free parameters, so the values can drive the
    fixed-parameter generator directly.
    """
    return {name: _family_default_params(name) for name in seed_model_names(seed_models_dir)}


def resolve_generating_params(
    spec: Any,
    seed_models_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Turn a config's ``generating_models`` spec into per-model fixed params.

    Accepts three shapes:
      * ``None`` — every seed model, each with its family defaults.
      * a list of names — those models, each with family defaults.
      * a dict ``name -> params`` — explicit params; a ``None``/empty value for
        a model falls back to that family's defaults.
    """
    if spec is None:
        return default_generating_params(seed_models_dir)
    if isinstance(spec, (list, tuple)):
        return {name: _family_default_params(name) for name in spec}
    if isinstance(spec, Mapping):
        return {
            name: (dict(params) if params else _family_default_params(name))
            for name, params in spec.items()
        }
    raise TypeError(
        f"generating_models must be null, a list of names, or a name->params "
        f"mapping; got {type(spec).__name__}."
    )


def feature_rows(stimuli: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """Featurize each (sequence_a, sequence_b) stimulus into a full feature row.

    The row carries every feature column any seed model reads, plus the raw
    sequences for reference. A dummy ``chose_left`` is included because the
    PyMC models declare it as a ``pm.Data`` input (its value is irrelevant to
    the ``p_left`` computation).
    """
    if not stimuli:
        raise ValueError("No stimuli provided.")
    rows: List[Dict[str, Any]] = []
    for stim in stimuli:
        seq_a, seq_b = stim["sequence_a"], stim["sequence_b"]
        row: Dict[str, Any] = {"sequence_a": seq_a, "sequence_b": seq_b}
        row.update(featurize_stimulus(seq_a, seq_b))
        row["chose_left"] = 0  # dummy; unused for p_left
        rows.append(row)
    return rows


def _require_exact_params(model, params: Mapping[str, float]) -> None:
    """Fail loudly unless `params` names exactly the model's free parameters.

    A missing parameter would leave that variable random under `pm.do`, silently
    turning a fixed-parameter generator into a prior sample.
    """
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

    Fixes every free parameter to its given value via `pm.do`, then evaluates
    the model's ``p_left`` deterministic on the featurized stimuli. Because all
    parameters are fixed, ``p_left`` is deterministic and the RNG seed only
    affects sampling overhead (kept for API symmetry).

    Returns an array of shape ``(n_stimuli,)``.
    """
    import pymc as pm

    model = load_pymc_model(model_name, models_dir)
    _require_exact_params(model, params)
    rows = feature_rows(stimuli)
    stim_data = make_stim_data(model, rows)

    with model:
        pm.set_data(stim_data)
    fixed = pm.do(model, dict(params))
    with fixed:
        prior = pm.sample_prior_predictive(
            draws=1, var_names=["p_left"], random_seed=seed
        )
    # shape (chain=1, draw=1, n_stim) -> (n_stim,)
    return np.asarray(prior.prior["p_left"].values).reshape(-1)


def p_left_model_family(
    model_name: str,
    stimuli: Sequence[Mapping[str, str]],
    params: Mapping[str, float],
) -> np.ndarray:
    """``p_left`` per stimulus from the pure-Python model family ``model_name``.

    Uses the family's ``predict_left`` — a different functional form than the
    PyMC adapter of the same name, so recovering it is a harder (more honest)
    test of the loop. Fails loudly unless ``params`` names exactly the family's
    parameters.
    """
    module = importlib.import_module(f"src.subjective_randomness.model_families.{model_name}")
    expected = set(module.DEFAULT_PARAMS)
    if set(params) != expected:
        missing = sorted(expected - set(params))
        extra = sorted(set(params) - expected)
        raise ValueError(
            f"Generating params must name exactly {model_name}'s parameters "
            f"{sorted(expected)}. Missing: {missing}. Unexpected: {extra}."
        )
    return np.array(
        [module.predict_left(stim, dict(params)) for stim in stimuli], dtype="float64"
    )


def generate_responses(
    model_name: str,
    models_dir: Path,
    stimuli: Sequence[Mapping[str, str]],
    params: Mapping[str, float],
    n_participants: int,
    *,
    seed: int = 0,
    generator: str = "pymc",
) -> List[Dict[str, Any]]:
    """Generate synthetic responses from a seed model with fixed parameters.

    Computes ``p_left`` per stimulus, then draws ``chose_left ~ Bernoulli(p_left)``
    for each of ``n_participants`` over every stimulus. Each returned row carries
    the full feature set (so any seed model can be fit on it), the raw sequences,
    ``chose_left``, and the name of the generating model.

    ``generator`` selects the data-generating process:
      * ``"pymc"`` (default) — the PyMC seed model with fixed parameters, so the
        true model and one candidate are literally identical.
      * ``"model_family"`` — the pure-Python ``model_families`` family of the
        same name, whose functional form differs from the PyMC fit (a harder
        recovery test).
    """
    if n_participants < 1:
        raise ValueError(f"n_participants must be >= 1, got {n_participants}.")

    stim_feature_rows = feature_rows(stimuli)
    if generator == "pymc":
        p_left = p_left_fixed_params(model_name, models_dir, stimuli, params, seed=seed)
    elif generator == "model_family":
        p_left = p_left_model_family(model_name, stimuli, params)
    else:
        raise ValueError(
            f"Unknown generator {generator!r}; expected 'pymc' or 'model_family'."
        )
    rng = np.random.default_rng(seed)

    rows: List[Dict[str, Any]] = []
    for participant in range(n_participants):
        draws = rng.random(len(stim_feature_rows)) < p_left
        for trial_index, (feat, chose_left) in enumerate(zip(stim_feature_rows, draws)):
            row = {k: v for k, v in feat.items() if k != "chose_left"}
            row["participant_id"] = participant
            row["trial_index"] = trial_index
            row["chose_left"] = int(chose_left)
            row["generating_model"] = model_name
            rows.append(row)
    return rows


def write_responses_csv(rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    """Write generated response rows to a featurized responses CSV.

    The column set is taken from the first row (every row shares it). The inner
    loop reads only the columns each model needs via its ``pm.Data`` inputs.
    """
    if not rows:
        raise ValueError("No response rows to write.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_closed_ended_recovery(
    stimuli: Sequence[Mapping[str, str]],
    seed_models_dir: Path,
    *,
    generating_params: Optional[Mapping[str, Mapping[str, float]]] = None,
    n_participants: int = 30,
    results_root: Path,
    fit_kwargs: Optional[Mapping[str, Any]] = None,
    cache_dir: Optional[Path] = None,
    seed: int = 0,
    generator: str = "pymc",
) -> Dict[str, Any]:
    """Run closed-ended model recovery over the seed set.

    For each generating model, synthesize data from it (fixed parameters), then
    run the inner model loop on the *closed* seed set (``max_iterations=0`` — no
    agent-proposed candidates) and record the recovered posterior over models.

    ``generating_params`` maps model name -> fixed parameter dict; it selects
    which models generate data (defaults to every seed model with its family's
    ``DEFAULT_PARAMS``). The recovered model set is always the full seed set.
    ``generator`` chooses the data source (``"pymc"`` seed model vs. the
    pure-Python ``"model_family"``); see :func:`generate_responses`.

    Returns a confusion-matrix-shaped dict: a ``generating`` list with one entry
    per true model, each carrying the posterior and ELPD-LOO over every seed model.
    """
    seed_models_dir = Path(seed_models_dir)
    results_root = Path(results_root)
    fit_kwargs = dict(fit_kwargs or {})
    if generating_params is None:
        generating_params = default_generating_params(seed_models_dir)

    seed_models = seed_model_names(seed_models_dir)
    generating: List[Dict[str, Any]] = []
    for gen_model, params in generating_params.items():
        rows = generate_responses(
            gen_model,
            seed_models_dir,
            stimuli,
            params,
            n_participants=n_participants,
            seed=seed,
            generator=generator,
        )
        model_root = results_root / gen_model
        responses_path = model_root / "responses.csv"
        write_responses_csv(rows, responses_path)

        loop_result = run_pymc_inner_loop(
            responses_path,
            model_root / "loop",
            seed_models_dir=seed_models_dir,
            max_iterations=0,
            cache_dir=cache_dir,
            fit_kwargs=fit_kwargs,
        )
        generating.append(
            {
                "generating_model": gen_model,
                "params": dict(params),
                "best_model": loop_result["best_model"],
                "recovered_correct": loop_result["best_model"] == gen_model,
                "posteriors": loop_result["posteriors"],
                "elpd_loo": loop_result["elpd_loo"],
                # PSIS-LOO distinguishability table (elpd_diff/dse per model),
                # so analysis can tell a clear recovery from a near-tie.
                "comparison": loop_result.get("comparison", {}),
            }
        )

    return {
        "seed_models": seed_models,
        "n_participants": n_participants,
        "n_stimuli": len(stimuli),
        "generator": generator,
        "fit_kwargs": fit_kwargs,
        "generating": generating,
    }


def run_recovery_from_config(
    config: Mapping[str, Any],
    config_path: Path,
    results_root: Path,
    *,
    n_participants_override: Optional[int] = None,
    fit_overrides: Optional[Mapping[str, Any]] = None,
    seed_override: Optional[int] = None,
    generator_override: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run closed-ended recovery from a config dict (as the CLI loads it).

    Config keys:
        seed_models_dir    directory with the seed model set + manifest
        stimuli_path       JSON list of {sequence_a, sequence_b}
        n_participants     synthetic participants per generating model
        generating_models  null | [names] | {name: params|null}  (default: all)
        generator          "pymc" (default) | "model_family"
        fit                MCMC kwargs (draws/tune/chains/...)
        seed               RNG seed for the synthetic choices
    Paths are resolved relative to the repo root, then the config's directory.
    """
    seed_models_dir = resolve_path(config["seed_models_dir"], config_path)
    stimuli = load_stimuli(resolve_path(config["stimuli_path"], config_path))
    n_participants = n_participants_override or int(config.get("n_participants", 30))
    generating_params = resolve_generating_params(
        config.get("generating_models"), seed_models_dir
    )
    fit_kwargs = {**dict(config.get("fit", {})), **dict(fit_overrides or {})}
    seed = seed_override if seed_override is not None else int(config.get("seed", 0))
    generator = generator_override or config.get("generator", "pymc")

    return run_closed_ended_recovery(
        stimuli,
        seed_models_dir,
        generating_params=generating_params,
        n_participants=n_participants,
        results_root=results_root,
        fit_kwargs=fit_kwargs,
        cache_dir=cache_dir,
        seed=seed,
        generator=generator,
    )


# Column order for the tidy confusion CSV written from `confusion_tidy_rows`.
CONFUSION_COLUMNS = [
    "generating_model",
    "recovered_model",
    "posterior",
    "elpd_loo",
    "is_true_model",
    "is_best_model",
]


def confusion_tidy_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flatten a recovery result into one row per (generating, recovered) cell.

    Long format ready for a confusion-matrix heatmap: the recovered posterior
    mass and ELPD-LOO the true model's data assigned to each seed model, plus
    flags marking the true model and the recovered best model.
    """
    rows: List[Dict[str, Any]] = []
    for record in result["generating"]:
        gen_model = record["generating_model"]
        best_model = record["best_model"]
        for recovered_model, posterior in record["posteriors"].items():
            rows.append(
                {
                    "generating_model": gen_model,
                    "recovered_model": recovered_model,
                    "posterior": posterior,
                    "elpd_loo": record["elpd_loo"][recovered_model],
                    "is_true_model": recovered_model == gen_model,
                    "is_best_model": recovered_model == best_model,
                }
            )
    return rows
