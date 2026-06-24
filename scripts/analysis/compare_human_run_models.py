"""CLI: how similar are the best-fitting models the three human runs arrived at?

Each live human run (``run1``/``run2``/``run3``) ends every experiment with a
Bayesian model posterior over candidate cognitive models. For a given experiment
we take the model(s) each run settled on — the single highest-posterior model, or
*both* models when two are statistically tied (e.g. run2/experiment3, where
``inner_loop_model`` and ``evidence_accumulation_per_run`` split the posterior
~0.49/0.46) — and ask how similar those models actually are *as theories of
behavior*.

We measure similarity exactly the way the model-recovery analysis does: fit each
model to the data it was selected on (the run's pooled responses, MCMC), predict
its population-level ``p_left`` over an exhaustive stimulus pool
(``enumerate_all_pairs`` — every H/T pair over the chosen sequence lengths,
cross-length pairs included), then take the RMSE between two models' prediction
vectors. Two models that encode the same behavioral theory predict near-identical
``p_left`` everywhere (RMSE ~ 0); genuinely different theories diverge.

Because every prediction is the *fitted* posterior-mean ``p_left`` (the user's
explicit choice), the RMSE reflects the models as the runs actually arrived at
them — structure *and* fitted parameters — not their default-parameter shapes.

Model names like ``iter1_candidate0`` are auto-generated per run, so the same
name in two runs is two *different* models; every label is prefixed with its run
(``run1/iter1_candidate0``) to keep them distinct.

The script fails loudly when a run has no ``model_posterior.json`` for the
experiment, when a selected model's source or responses file is missing, or when
a model is hierarchical (its ``p_left`` is per-participant and has no single
population-level vector to compare here).

Usage:
    # Default: experiment3 (each run's final state), pool over lengths 6 & 8.
    uv run python scripts/analysis/compare_human_run_models.py

    # Another experiment / different stimulus lengths / output dir.
    uv run python scripts/analysis/compare_human_run_models.py \\
        --experiment experiment2 --lengths 6 8 --out-dir /tmp/figs
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402

DEFAULT_RUNS_ROOT = Path("data/results/human_experiment")
PROJECT_ID = "subjective_randomness"

# Live-run MCMC settings (scripts/outer_loop_live/full_run.yaml) — the fits that
# selected these models. We refit with the same settings so the predictions
# reflect the models as the runs arrived at them.
DEFAULT_DRAWS = 3000
DEFAULT_TUNE = 2000
DEFAULT_CHAINS = 4


@dataclass
class Args:
    """Pairwise RMSE between the best-fitting models of the three human runs."""

    runs_root: Path = DEFAULT_RUNS_ROOT
    """Directory holding ``run<r>/subjective_randomness/experiment<e>/model_loop``."""
    experiment: str = "experiment3"
    """Which experiment's winning models to compare (default: each run's final state)."""
    lengths: Tuple[int, ...] = (6, 8)
    """Sequence lengths whose every H/T pair forms the eval pool (model-recovery default)."""
    tie_ratio: float = 0.5
    """A second model counts as tied (and is included) if its posterior is at least
    this fraction of the run's top model's posterior."""
    out_dir: Optional[Path] = None
    """Where to write the matrix + long-form CSVs (default: ``--runs-root``)."""
    cache_dir: Optional[Path] = None
    """MCMC fit cache (default: ``<out_dir>/model_similarity_cache``)."""
    draws: int = DEFAULT_DRAWS
    tune: int = DEFAULT_TUNE
    chains: int = DEFAULT_CHAINS
    cores: int = DEFAULT_CHAINS
    predict_max_draws: int = 500
    """Thin the posterior to at most this many draws for the prediction pass, so the
    (draws × n_stimuli) array stays bounded over the large exhaustive pool; the
    posterior-mean ``p_left`` is essentially unchanged by the thinning."""


# ─────────────────────────────────────────────
# Pure helpers (unit-tested)
# ─────────────────────────────────────────────


def model_label(run: str, name: str) -> str:
    """Run-prefixed model label, so identically-named models stay distinct."""
    return f"{run}/{name}"


def select_winning_models(
    posteriors: Mapping[str, float], *, tie_ratio: float
) -> List[str]:
    """The model(s) a run settled on: the top model plus any tied with it.

    Returns the highest-posterior model, followed by every other model whose
    posterior is at least ``tie_ratio`` times the top model's posterior, in
    descending-posterior order. With a dominant winner this is a single model;
    with a near-even split (two models sharing the posterior mass) it is both.
    Fails loudly if the posterior is empty or has no positive mass.
    """
    if not posteriors:
        raise ValueError("Empty posterior: no models to select from.")
    ranked = sorted(posteriors.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_mass = ranked[0]
    if top_mass <= 0.0:
        raise ValueError(
            f"Posterior has no positive mass (max is {top_mass}); cannot pick a winner."
        )
    threshold = tie_ratio * top_mass
    return [name for name, mass in ranked if mass >= threshold]


def pairwise_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two prediction vectors: ``sqrt(mean((a - b) ** 2))``."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def pairwise_rmse_matrix(
    predictions: Mapping[str, np.ndarray],
) -> Tuple[List[str], np.ndarray]:
    """Symmetric RMSE matrix over the prediction vectors.

    ``predictions`` maps each model label to its ``p_left`` vector; all vectors
    must share length (the same stimulus pool). Returns the labels (insertion
    order) and an ``(n, n)`` matrix whose ``[i, j]`` is the RMSE between models
    ``i`` and ``j`` (0 on the diagonal). Needs at least two models with
    equal-length vectors.
    """
    labels = list(predictions)
    if len(labels) < 2:
        raise ValueError(
            f"Need at least two models to compare; got {len(labels)}: {labels}"
        )
    lengths = {label: len(np.asarray(v)) for label, v in predictions.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Prediction vectors must share length (same stimulus pool); got {lengths}"
        )
    n = len(labels)
    mat = np.zeros((n, n), dtype="float64")
    for i in range(n):
        for j in range(i + 1, n):
            r = pairwise_rmse(predictions[labels[i]], predictions[labels[j]])
            mat[i, j] = mat[j, i] = r
    return labels, mat


def _posterior_path(runs_root: Path, run: str, experiment: str) -> Path:
    return (
        Path(runs_root)
        / run
        / PROJECT_ID
        / experiment
        / "model_loop"
        / "model_posterior.json"
    )


def resolve_winners(
    runs_root: Path, experiment: str, *, tie_ratio: float
) -> List[Tuple[str, str]]:
    """``(run, model_name)`` winners for ``experiment`` across every run dir.

    Discovers each ``run*`` directory under ``runs_root`` that holds a
    ``model_posterior.json`` for the experiment, and selects the winning model(s)
    per run via :func:`select_winning_models`. Fails loudly when no run has a
    posterior for the experiment.
    """
    runs_root = Path(runs_root)
    paths = sorted(runs_root.glob(f"run*/{PROJECT_ID}/{experiment}/model_loop/model_posterior.json"))
    if not paths:
        raise FileNotFoundError(
            f"No model_posterior.json under {runs_root}/run*/{PROJECT_ID}/"
            f"{experiment}/model_loop — nothing to compare."
        )
    winners: List[Tuple[str, str]] = []
    for path in paths:
        run = path.parts[len(runs_root.parts)]  # the run* directory name
        payload = json.loads(path.read_text(encoding="utf-8"))
        for name in select_winning_models(payload["posteriors"], tie_ratio=tie_ratio):
            winners.append((run, name))
    return winners


# ─────────────────────────────────────────────
# Fitted predictions (MCMC; not unit-tested)
# ─────────────────────────────────────────────


def fit_and_predict_p_left(
    run: str,
    name: str,
    runs_root: Path,
    experiment: str,
    eval_rows: Sequence[Mapping[str, object]],
    *,
    fit_kwargs: Mapping[str, int],
    cache_dir: Path,
    predict_max_draws: int,
) -> np.ndarray:
    """Fit ``run``'s ``name`` model on its own data and predict ``p_left`` on the pool.

    The model is fit (MCMC) on the run's pooled inner-loop responses for the
    experiment, then its posterior-mean ``p_left`` is predicted for every row in
    ``eval_rows``. Hierarchical models (a ``participant_id`` input) have no single
    population-level ``p_left`` vector to compare and raise here.
    """
    from src.models.pymc_inference import fit_model, make_stim_data, pm_data_inputs

    loop_dir = Path(runs_root) / run / PROJECT_ID / experiment / "model_loop"
    models_dir = loop_dir / "models"
    responses_path = loop_dir / "responses.csv"
    if not (models_dir / f"{name}.py").exists():
        raise FileNotFoundError(f"Model source not found: {models_dir / f'{name}.py'}")
    if not responses_path.exists():
        raise FileNotFoundError(f"Responses not found for {run}/{experiment}: {responses_path}")

    fitted = fit_model(
        name, models_dir, responses_path, cache_dir=cache_dir, **dict(fit_kwargs)
    )
    if "participant_id" in pm_data_inputs(fitted.model):
        raise ValueError(
            f"{model_label(run, name)} is hierarchical (indexes participant_id); it "
            f"has no single population-level p_left vector to compare here."
        )
    stim_data = make_stim_data(fitted.model, list(eval_rows))
    preds = fitted.predict_p_left(stim_data, max_draws=predict_max_draws)
    return np.asarray(preds, dtype="float64")


def write_matrix_csv(labels: Sequence[str], matrix: np.ndarray, path: Path) -> None:
    """Write the square RMSE matrix with a leading label column/header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *(f"{v:.6f}" for v in row)])


def write_long_csv(labels: Sequence[str], matrix: np.ndarray, path: Path) -> None:
    """Write the upper-triangle pairwise RMSEs as ``model_a, model_b, rmse`` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_a", "model_b", "rmse"])
        n = len(labels)
        for i in range(n):
            for j in range(i + 1, n):
                writer.writerow([labels[i], labels[j], f"{matrix[i, j]:.6f}"])


def format_matrix(labels: Sequence[str], matrix: np.ndarray) -> str:
    """A readable fixed-width text rendering of the RMSE matrix for the console."""
    width = max(len(label) for label in labels)
    header = " " * (width + 2) + "  ".join(f"{label:>{width}}" for label in labels)
    lines = [header]
    for label, row in zip(labels, matrix):
        cells = "  ".join(f"{v:>{width}.4f}" for v in row)
        lines.append(f"{label:>{width}}  {cells}")
    return "\n".join(lines)


def main(args: Args) -> None:
    runs_root = resolve_path(args.runs_root)
    out_dir = resolve_path(args.out_dir) if args.out_dir is not None else runs_root
    cache_dir = (
        resolve_path(args.cache_dir)
        if args.cache_dir is not None
        else out_dir / "model_similarity_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    from src.subjective_randomness.model_recovery import feature_rows
    from src.subjective_randomness.stimulus_design import enumerate_all_pairs

    winners = resolve_winners(runs_root, args.experiment, tie_ratio=args.tie_ratio)
    print(f"Comparing {len(winners)} best-fitting model(s) for {args.experiment}:")
    for run, name in winners:
        print(f"  {model_label(run, name)}")

    pool = enumerate_all_pairs(args.lengths)
    eval_rows = feature_rows(pool)
    print(
        f"\nStimulus pool: {len(pool)} pairs over lengths {tuple(args.lengths)} "
        f"(enumerate_all_pairs)."
    )

    fit_kwargs = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "cores": args.cores,
    }
    predictions: Dict[str, np.ndarray] = {}
    for run, name in winners:
        label = model_label(run, name)
        print(f"\nFitting + predicting {label} ...", flush=True)
        predictions[label] = fit_and_predict_p_left(
            run,
            name,
            runs_root,
            args.experiment,
            eval_rows,
            fit_kwargs=fit_kwargs,
            cache_dir=cache_dir,
            predict_max_draws=args.predict_max_draws,
        )

    labels, matrix = pairwise_rmse_matrix(predictions)

    print("\nPairwise RMSE between fitted p_left predictions:\n")
    print(format_matrix(labels, matrix))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"human_run_model_rmse_{args.experiment}"
    matrix_path = out_dir / f"{stem}_matrix.csv"
    long_path = out_dir / f"{stem}_pairs.csv"
    write_matrix_csv(labels, matrix, matrix_path)
    write_long_csv(labels, matrix, long_path)
    print(f"\nWrote matrix to {matrix_path}")
    print(f"Wrote pairwise CSV to {long_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
