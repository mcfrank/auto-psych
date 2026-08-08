"""CLI: how many candidate stimuli can design-time EIG afford to score?

The outer loop currently conjectures ~30 candidate stimuli and annotates each
with prior-predictive EIG (``src.pipelines.outer_loop.eig.annotate``). That
path calls ``sample_prior_predictive`` once per model *per stimulus*, so its
cost is dominated by a fixed per-call overhead. This benchmark measures how far
the candidate count can scale by timing, on the same models and stimuli:

1. **per_stimulus** — the pipeline's ``annotate`` exactly as the design stage
   runs it, and
2. **batched** — one ``sample_prior_predictive`` per model for *all* candidates
   at once (``prior_predict_p_left_batch``), which pays the per-call overhead
   once per model total.

Before timing anything the two paths are compared on the same candidates and
the run aborts if their EIG values disagree (same seed → identical prior draws
→ identical values, so any gap is a bug). Wall times per candidate-count are
then fitted with a line per method, and the fits are inverted into the maximum
candidate count affordable within each time budget — also compared against the
exhaustive pair universe (every distinct pair of H/T sequences over the chosen
lengths, cross-length pairs included, as in ``enumerate_all_pairs``).

Outputs (in ``--out-dir``): ``eig_scaling_timings.csv`` (one row per timed
run), ``eig_scaling_summary.json`` (fits, projections, universe size), and
``eig_scaling.png`` (timing curves).

Usage:
    # Default: the 4 live seed models, pipeline n_samples=200, lengths 4-8.
    uv run python scripts/analysis/benchmark_eig_scaling.py

    # Another model set / bigger batched sizes / registry-weighted model prior.
    uv run python scripts/analysis/benchmark_eig_scaling.py \\
        --models-dir data/outer_loop/subjective_randomness/experiment1/cognitive_models \\
        --batched-sizes 1000 10000 50000 --registry path/to/model_registry.yaml
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tyro
from pyprojroot import here

REPO_ROOT = here()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.pymc_inference import (  # noqa: E402
    eig_from_prior_means,
    load_pymc_model_cached,
    make_stim_data,
    prior_predict_p_left_batch,
)
from src.pipelines.outer_loop.eig import annotate  # noqa: E402
from src.subjective_randomness.features import featurize_stimulus  # noqa: E402
from src.subjective_randomness.stimulus_design import (  # noqa: E402
    generate_candidate_pool,
)

# First two slots of the categorical palette (fixed order, colorblind-safe).
METHOD_COLORS = {"per_stimulus": "#2a78d6", "batched": "#eb6834"}

MAX_ENUMERABLE_LENGTH = 12  # mirrors the enumerate_all_pairs cap


def pair_universe_size(lengths: Sequence[int]) -> int:
    """Number of distinct unordered sequence pairs over the given lengths.

    Closed form for ``len(enumerate_all_pairs(lengths))``: the pool is the
    union of all H/T sequences of each length (cross-length pairs included),
    so the universe is C(pool, 2).
    """
    unique = sorted(set(lengths))
    if not unique:
        raise ValueError("lengths must be non-empty.")
    if any(length < 1 for length in unique):
        raise ValueError(f"Sequence lengths must be >= 1, got {tuple(lengths)}.")
    if any(length > MAX_ENUMERABLE_LENGTH for length in unique):
        raise ValueError(
            f"Sequence lengths are capped at {MAX_ENUMERABLE_LENGTH} to bound enumeration."
        )
    pool = sum(2**length for length in unique)
    return math.comb(pool, 2)


def fit_linear_seconds(
    sizes: Sequence[int], seconds: Sequence[float]
) -> Tuple[float, float]:
    """Least-squares fit ``seconds ~ intercept + slope * n_stimuli``.

    Returns ``(intercept, slope)``. Needs at least two distinct sizes; the
    intercept captures fixed per-run overhead (manifest read, model screen,
    sampler setup), the slope is the marginal cost per candidate.
    """
    if len(sizes) != len(seconds):
        raise ValueError(f"Got {len(sizes)} sizes but {len(seconds)} timings.")
    if len(set(sizes)) < 2:
        raise ValueError("Need timings at >= 2 distinct sizes to fit a line.")
    design = np.column_stack([np.ones(len(sizes)), np.asarray(sizes, dtype=float)])
    coef, *_ = np.linalg.lstsq(design, np.asarray(seconds, dtype=float), rcond=None)
    return float(coef[0]), float(coef[1])


def effective_time_model(
    sizes: Sequence[int], seconds: Sequence[float]
) -> Tuple[float, float, bool]:
    """Time model ``(intercept, slope, slope_resolved)`` for one method.

    Uses the least-squares line when its slope is positive. When the measured
    timings are flat (marginal cost below measurement noise, so the fitted
    slope is <= 0), extrapolating from that fit would be meaningless; fall back
    to the conservative throughput at the largest measured size — all overhead
    folded into the per-stimulus cost — which makes every projection a lower
    bound rather than a failure.
    """
    intercept, slope = fit_linear_seconds(sizes, seconds)
    if slope > 0:
        return intercept, slope, True
    i_largest = max(range(len(sizes)), key=lambda i: sizes[i])
    return 0.0, seconds[i_largest] / sizes[i_largest], False


def max_candidates_within(budget_s: float, *, intercept: float, slope: float) -> int:
    """Largest candidate count whose predicted wall time fits in ``budget_s``."""
    if slope <= 0:
        raise ValueError(f"Fitted slope must be positive, got {slope}.")
    return max(0, math.floor((budget_s - intercept) / slope))


@dataclass(frozen=True)
class TimingRecord:
    method: str
    n_stimuli: int
    n_models: int
    n_samples: int
    seconds: float
    seconds_per_stimulus: float


@dataclass(frozen=True)
class BenchmarkResult:
    records: List[TimingRecord]
    max_abs_eig_diff: float
    universe_size: int
    fits: Dict[str, Tuple[float, float]]  # method -> (intercept, slope)
    slope_resolved: Dict[str, bool]  # False -> conservative throughput fallback
    projections: Dict[str, Dict[float, int]]  # method -> budget_s -> max n
    predicted_universe_seconds: Dict[str, float]
    model_names: List[str]
    warmup_seconds: float


def _featurized_candidates(
    n_pairs: int, lengths: Sequence[int], seed: int
) -> List[Dict[str, Any]]:
    """Sample candidate pairs and merge in their feature columns upfront.

    Pre-featurizing keeps the timed comparison about the sampling paths, not
    the (negligible, identical) featurization; ``annotate`` accepts rows that
    already carry feature columns.
    """
    pool = generate_candidate_pool(n_pairs, lengths=tuple(lengths), seed=seed)
    rows: List[Dict[str, Any]] = []
    for cand in pool:
        row: Dict[str, Any] = dict(cand)
        row.update(featurize_stimulus(cand["sequence_a"], cand["sequence_b"]))
        row["chose_left"] = 0  # dummy; ignored for prior-predictive p_left
        rows.append(row)
    return rows


def _usable_model_names(
    models_dir: Path, probe_row: Dict[str, Any]
) -> List[str]:
    """Manifest model names that can be evaluated on a bare stimulus row.

    Same screen ``annotate`` applies: models needing columns stimuli never
    carry (e.g. participant_id) are dropped loudly; failing only if none load.
    """
    import yaml

    from src.models.theorist.loader import get_model_names_from_manifest

    manifest_path = Path(models_dir) / "models_manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"models_manifest.yaml not found at {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    names = get_model_names_from_manifest(manifest, Path(models_dir))
    if not names:
        raise ValueError(f"No loadable models found in {models_dir}")

    usable = []
    for name in names:
        try:
            make_stim_data(load_pymc_model_cached(name, Path(models_dir)), [probe_row])
        except Exception as e:  # noqa: BLE001 — unbindable model can't be scored
            print(f"  [drop] model {name!r} cannot score a stimulus ({e}); skipping.")
            continue
        usable.append(name)
    if not usable:
        raise ValueError(f"No model in {models_dir} can be evaluated on a stimulus.")
    return usable


def eig_batched(
    rows: List[Dict[str, Any]],
    model_names: List[str],
    models_dir: Path,
    model_weights: Optional[Dict[str, float]] = None,
    *,
    n_samples: int,
    seed: int,
) -> List[float]:
    """EIG for every row from one batched prior-predictive pass per model."""
    means = prior_predict_p_left_batch(
        model_names, models_dir, rows, n_samples=n_samples, seed=seed
    )
    return [
        eig_from_prior_means({m: float(means[m][i]) for m in means}, model_weights)
        for i in range(len(rows))
    ]


def _check_agreement(
    rows: List[Dict[str, Any]],
    model_names: List[str],
    models_dir: Path,
    model_weights: Optional[Dict[str, float]],
    *,
    n_samples: int,
    seed: int,
    tol: float,
) -> float:
    """Max |EIG difference| between the pipeline path and the batched path.

    Raises if the paths disagree beyond ``tol``: with the same seed the prior
    parameter draws are identical, so any real gap means one path is wrong and
    every downstream timing claim would be about a broken implementation.
    """
    annotated = annotate(
        list(rows),
        models_dir,
        registry_path=None,
        featurize_path=None,
        n_samples=n_samples,
        seed=seed,
    )
    per_stim = {
        (a["sequence_a"], a["sequence_b"]): a["eig"] for a in annotated
    }
    batched = eig_batched(
        rows, model_names, models_dir, model_weights, n_samples=n_samples, seed=seed
    )
    diffs = [
        abs(per_stim[(r["sequence_a"], r["sequence_b"])] - e)
        for r, e in zip(rows, batched)
    ]
    max_diff = max(diffs)
    # annotate() rounds EIG to 6 decimals on output, so agreement below the
    # rounding grain is the strongest claim the comparison can make.
    if max_diff > max(tol, 5e-7):
        raise RuntimeError(
            f"Batched EIG disagrees with the pipeline path (max |diff| = "
            f"{max_diff:.2e} > {tol:.0e}); refusing to benchmark a wrong implementation."
        )
    return max_diff


def run_benchmark(
    models_dir: Path,
    *,
    per_stim_sizes: Sequence[int],
    batched_sizes: Sequence[int],
    n_samples: int,
    seed: int,
    lengths: Sequence[int],
    budgets_s: Sequence[float],
    agreement_n: int,
    out_dir: Path,
    model_weights: Optional[Dict[str, float]] = None,
    agreement_tol: float = 1e-6,
) -> BenchmarkResult:
    """Time both EIG paths, project budgets, and write all artifacts."""
    models_dir = Path(models_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_max = max([*per_stim_sizes, *batched_sizes, agreement_n])
    rows = _featurized_candidates(n_max, lengths, seed)
    model_names = _usable_model_names(models_dir, rows[0])

    # Warm up both code paths (model build + PyTensor compilation) so timings
    # reflect the steady state a real annotation run spends its time in; the
    # one-off cost is reported separately.
    t0 = time.perf_counter()
    annotate([dict(rows[0])], models_dir, featurize_path=None, n_samples=n_samples, seed=seed)
    eig_batched(rows[:2], model_names, models_dir, n_samples=n_samples, seed=seed)
    warmup_seconds = time.perf_counter() - t0

    print(f"Models: {model_names} (warmup {warmup_seconds:.1f}s)")

    max_abs_eig_diff = _check_agreement(
        rows[:agreement_n],
        model_names,
        models_dir,
        model_weights,
        n_samples=n_samples,
        seed=seed,
        tol=agreement_tol,
    )
    print(f"Agreement on {agreement_n} candidates: max |diff| = {max_abs_eig_diff:.2e}")

    records: List[TimingRecord] = []

    def record(method: str, n: int, seconds: float) -> None:
        records.append(
            TimingRecord(
                method=method,
                n_stimuli=n,
                n_models=len(model_names),
                n_samples=n_samples,
                seconds=seconds,
                seconds_per_stimulus=seconds / n,
            )
        )
        print(f"  {method:>12s}  n={n:>6d}  {seconds:8.2f}s  ({seconds / n:.3f}s/stim)")

    print("Timing the pipeline per-stimulus path:")
    for n in per_stim_sizes:
        t0 = time.perf_counter()
        annotate(
            [dict(r) for r in rows[:n]],
            models_dir,
            registry_path=None,
            featurize_path=None,
            n_samples=n_samples,
            seed=seed,
        )
        record("per_stimulus", n, time.perf_counter() - t0)

    print("Timing the batched path:")
    for n in batched_sizes:
        t0 = time.perf_counter()
        eig_batched(
            rows[:n], model_names, models_dir, model_weights,
            n_samples=n_samples, seed=seed,
        )
        record("batched", n, time.perf_counter() - t0)

    universe = pair_universe_size(lengths)
    fits: Dict[str, Tuple[float, float]] = {}
    slope_resolved: Dict[str, bool] = {}
    projections: Dict[str, Dict[float, int]] = {}
    predicted_universe_seconds: Dict[str, float] = {}
    for method in ("per_stimulus", "batched"):
        method_recs = [r for r in records if r.method == method]
        intercept, slope, resolved = effective_time_model(
            [r.n_stimuli for r in method_recs], [r.seconds for r in method_recs]
        )
        if not resolved:
            print(
                f"  note: {method} marginal cost is below measurement noise at "
                "these sizes; projections are conservative throughput bounds."
            )
        fits[method] = (intercept, slope)
        slope_resolved[method] = resolved
        projections[method] = {
            float(b): max_candidates_within(float(b), intercept=intercept, slope=slope)
            for b in budgets_s
        }
        predicted_universe_seconds[method] = intercept + slope * universe

    result = BenchmarkResult(
        records=records,
        max_abs_eig_diff=max_abs_eig_diff,
        universe_size=universe,
        fits=fits,
        slope_resolved=slope_resolved,
        projections=projections,
        predicted_universe_seconds=predicted_universe_seconds,
        model_names=model_names,
        warmup_seconds=warmup_seconds,
    )
    _write_artifacts(result, lengths, out_dir)
    return result


def _write_artifacts(
    result: BenchmarkResult, lengths: Sequence[int], out_dir: Path
) -> None:
    csv_path = out_dir / "eig_scaling_timings.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result.records[0]).keys()))
        writer.writeheader()
        for rec in result.records:
            writer.writerow(asdict(rec))

    summary = {
        "model_names": result.model_names,
        "n_models": len(result.model_names),
        "n_samples": result.records[0].n_samples,
        "lengths": list(lengths),
        "universe_size": result.universe_size,
        "max_abs_eig_diff": result.max_abs_eig_diff,
        "warmup_seconds": result.warmup_seconds,
        "fits": {
            m: {
                "intercept_s": i,
                "slope_s_per_stimulus": s,
                "slope_resolved": result.slope_resolved[m],
            }
            for m, (i, s) in result.fits.items()
        },
        "projections_max_candidates": {
            m: {f"{b:g}s": n for b, n in by_budget.items()}
            for m, by_budget in result.projections.items()
        },
        "predicted_universe_seconds": result.predicted_universe_seconds,
    }
    (out_dir / "eig_scaling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot(result, out_dir / "eig_scaling.png")
    print(f"Wrote {csv_path}, summary JSON, and plot to {out_dir}")


def _plot(result: BenchmarkResult, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, (intercept, slope) in result.fits.items():
        recs = sorted(
            (r for r in result.records if r.method == method),
            key=lambda r: r.n_stimuli,
        )
        ns = [r.n_stimuli for r in recs]
        secs = [r.seconds for r in recs]
        color = METHOD_COLORS[method]
        ax.plot(ns, secs, "o", color=color, markersize=8, label=method)
        # Fitted line, extended one decade past the largest measurement.
        grid = np.geomspace(min(ns), max(ns) * 10, 50)
        ax.plot(grid, intercept + slope * grid, "-", color=color, linewidth=2, alpha=0.7)
        ax.annotate(
            f"{slope * 1000:.1f} ms/stim",
            (ns[-1], secs[-1]),
            textcoords="offset points",
            xytext=(8, -4),
            color="#444444",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of candidate stimuli")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(
        f"Design-time EIG cost — {len(result.model_names)} models, "
        f"{result.records[0].n_samples} prior draws"
    )
    ax.grid(True, which="both", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@dataclass
class Args:
    """Benchmark how many candidate stimuli design-time EIG can afford."""

    models_dir: Path = (
        REPO_ROOT / "src/pipelines/outer_loop/projects/subjective_randomness/seed_models"
    )
    """cognitive_models directory (models_manifest.yaml + PyMC .py files)."""
    out_dir: Path = REPO_ROOT / "data/analysis/eig_scaling"
    """Where the CSV / JSON / plot artifacts are written."""
    registry: Optional[Path] = None
    """Optional model_registry.yaml for a weighted model prior (uniform if omitted)."""
    per_stim_sizes: Tuple[int, ...] = (5, 10, 20, 40)
    """Candidate counts at which to time the pipeline's per-stimulus path."""
    batched_sizes: Tuple[int, ...] = (100, 500, 2000, 10000)
    """Candidate counts at which to time the batched path."""
    n_samples: int = 200
    """Prior-predictive draws per model (pipeline default: 200)."""
    seed: int = 42
    """Seed for candidate sampling and prior-predictive draws."""
    lengths: Tuple[int, ...] = (4, 5, 6, 7, 8)
    """Sequence lengths defining the candidate pool and the pair universe."""
    budgets_s: Tuple[float, ...] = (60.0, 600.0, 3600.0)
    """Time budgets (seconds) to translate into max candidate counts."""
    agreement_n: int = 16
    """Candidates used for the batched-vs-pipeline agreement check."""


def main(args: Args) -> None:
    model_weights: Optional[Dict[str, float]] = None
    if args.registry is not None:
        import yaml

        reg = yaml.safe_load(args.registry.read_text(encoding="utf-8")) or {}
        model_weights = reg.get("theories", {}) or None
        if model_weights is None:
            raise ValueError(f"{args.registry} has no 'theories' weights.")

    result = run_benchmark(
        args.models_dir,
        per_stim_sizes=args.per_stim_sizes,
        batched_sizes=args.batched_sizes,
        n_samples=args.n_samples,
        seed=args.seed,
        lengths=args.lengths,
        budgets_s=args.budgets_s,
        agreement_n=args.agreement_n,
        out_dir=args.out_dir,
        model_weights=model_weights,
    )

    print("\n--- How many candidates can we afford? ---")
    for method, (intercept, slope) in result.fits.items():
        print(f"{method}: {slope * 1000:.1f} ms/stimulus (+{intercept:.1f}s overhead)")
        for budget, n in result.projections[method].items():
            print(f"  {budget:>7.0f}s budget -> {n:>10,d} candidates")
    print(
        f"Exhaustive universe over lengths {list(args.lengths)}: "
        f"{result.universe_size:,d} pairs"
    )
    for method, secs in result.predicted_universe_seconds.items():
        print(f"  predicted exhaustive time ({method}): {secs / 3600:.2f} h")


if __name__ == "__main__":
    main(tyro.cli(Args))
