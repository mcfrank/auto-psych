"""CLI: how expensive is greedy joint-EIG stimulus-set selection at scale?

Batched per-draw scoring (``prior_predict_p_left_draws``) made it cheap to
score every candidate stimulus; this benchmark times the *selection* stage that
follows — ``select_n_joint_eig``, which greedily picks the N stimuli with
maximal joint EIG about model identity. For each pool size it times four
stages on the same models:

1. **featurize** — computing feature columns for the pool,
2. **score** — one batched prior-predictive pass per model (per-draw p_left),
3. **select_exact** — greedy with full gain re-scoring each step (the default),
4. **select_lazy** — CELF lazy greedy (approximate under synergy).

Both selections are re-estimated on fresh Monte Carlo scenarios so the
exact-vs-lazy quality gap is measured alongside the speed gap. With
``--exhaustive`` (default) the largest pool is the *entire* pair universe over
the chosen lengths (``enumerate_all_pairs``).

Outputs (in ``--out-dir``): ``joint_eig_selection_timings.csv``,
``joint_eig_selection_summary.json``, and ``joint_eig_selection.png``.

Usage:
    # Default: 4 hero-run seed models, k=32, pools up to the full universe.
    uv run python scripts/analysis/benchmark_joint_eig_selection.py

    # Smaller sweep, more scenarios:
    uv run python scripts/analysis/benchmark_joint_eig_selection.py \\
        --pool-sizes 2000 10000 --no-exhaustive --n-scenarios 4000
"""

from __future__ import annotations

import csv
import json
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

from src.models.eig_selection import (  # noqa: E402
    estimate_joint_eig,
    select_n_joint_eig,
)
from src.models.pymc_inference import (  # noqa: E402
    load_pymc_model_cached,
    make_stim_data,
    prior_predict_p_left_draws,
)
from src.subjective_randomness.features import featurize_stimulus  # noqa: E402
from src.subjective_randomness.stimulus_design import (  # noqa: E402
    enumerate_all_pairs,
    generate_candidate_pool,
)

# Categorical palette slots 1-3 (fixed order, colorblind-safe).
STAGE_COLORS = {
    "score": "#2a78d6",
    "select_exact": "#eb6834",
    "select_lazy": "#1baf7a",
}


@dataclass(frozen=True)
class TimingRecord:
    stage: str  # featurize | score | select_exact | select_lazy
    n_pool: int
    n_select: int
    n_models: int
    n_scenarios: int
    seconds: float


@dataclass(frozen=True)
class SelectionBenchmarkResult:
    records: List[TimingRecord]
    # pool size -> {exact_indices, exact_joint_eig_bits, lazy_indices,
    #               lazy_joint_eig_bits} with joint EIG from fresh scenarios.
    selections: Dict[int, Dict[str, Any]]
    model_names: List[str]


def _usable_model_names(models_dir: Path, probe_row: Dict[str, Any]) -> List[str]:
    """Manifest models evaluable on a bare stimulus row (same screen as eig.py)."""
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


def _featurize(pool: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cand in pool:
        row: Dict[str, Any] = dict(cand)
        row.update(featurize_stimulus(cand["sequence_a"], cand["sequence_b"]))
        row["chose_left"] = 0  # dummy; ignored for prior-predictive p_left
        rows.append(row)
    return rows


def run_benchmark(
    models_dir: Path,
    *,
    pool_sizes: Sequence[int],
    exhaustive: bool,
    lengths: Sequence[int],
    n_select: int,
    n_scenarios: int,
    n_samples: int,
    seed: int,
    out_dir: Path,
    model_weights: Optional[Dict[str, float]] = None,
    n_random_baseline: int = 20,
) -> SelectionBenchmarkResult:
    """Time featurize/score/select for each pool size and write artifacts."""
    models_dir = Path(models_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pools: List[Tuple[int, List[Dict[str, str]]]] = []
    for n in pool_sizes:
        pools.append((n, generate_candidate_pool(n, lengths=tuple(lengths), seed=seed)))
    if exhaustive:
        universe = enumerate_all_pairs(list(lengths))
        pools.append((len(universe), universe))

    probe = _featurize(pools[0][1][:1])[0]
    model_names = _usable_model_names(models_dir, probe)
    print(f"Models: {model_names}; selecting {n_select} per pool")

    records: List[TimingRecord] = []
    selections: Dict[int, Dict[str, Any]] = {}

    def record(stage: str, n_pool: int, seconds: float) -> None:
        records.append(
            TimingRecord(
                stage=stage,
                n_pool=n_pool,
                n_select=n_select,
                n_models=len(model_names),
                n_scenarios=n_scenarios,
                seconds=seconds,
            )
        )
        print(f"  {stage:>12s}  pool={n_pool:>7,d}  {seconds:8.2f}s")

    for n_pool, pool in pools:
        print(f"Pool of {n_pool:,d} candidates:")
        t0 = time.perf_counter()
        rows = _featurize(pool)
        record("featurize", n_pool, time.perf_counter() - t0)

        t0 = time.perf_counter()
        draws = prior_predict_p_left_draws(
            model_names, models_dir, rows, n_samples=n_samples, seed=seed
        )
        record("score", n_pool, time.perf_counter() - t0)

        chosen: Dict[str, Any] = {}
        for label, lazy in (("exact", False), ("lazy", True)):
            t0 = time.perf_counter()
            sel = select_n_joint_eig(
                draws,
                n_select,
                model_weights=model_weights,
                n_scenarios=n_scenarios,
                seed=seed,
                lazy=lazy,
            )
            record(f"select_{label}", n_pool, time.perf_counter() - t0)
            fresh = estimate_joint_eig(
                draws,
                sel.indices,
                model_weights=model_weights,
                n_scenarios=max(n_scenarios, 2000),
                seed=seed + 1,
            )
            chosen[f"{label}_indices"] = sel.indices
            chosen[f"{label}_joint_eig_bits"] = fresh

        # Random baseline: joint EIG of uniformly random k-sets from the same
        # pool, estimated with the same fresh scenarios as the optimized sets.
        baseline_rng = np.random.default_rng(seed + 12345)
        baseline_eigs = [
            estimate_joint_eig(
                draws,
                baseline_rng.choice(n_pool, size=n_select, replace=False).tolist(),
                model_weights=model_weights,
                n_scenarios=max(n_scenarios, 2000),
                seed=seed + 1,
            )
            for _ in range(n_random_baseline)
        ]
        chosen["random_joint_eig_bits"] = {
            "mean": float(np.mean(baseline_eigs)),
            "min": float(np.min(baseline_eigs)),
            "max": float(np.max(baseline_eigs)),
        }
        print(
            f"  fresh joint EIG: exact={chosen['exact_joint_eig_bits']:.4f} "
            f"lazy={chosen['lazy_joint_eig_bits']:.4f} "
            f"random={chosen['random_joint_eig_bits']['mean']:.4f} "
            f"[{chosen['random_joint_eig_bits']['min']:.4f}, "
            f"{chosen['random_joint_eig_bits']['max']:.4f}] bits "
            f"(n={n_random_baseline})"
        )
        selections[n_pool] = chosen

    result = SelectionBenchmarkResult(
        records=records, selections=selections, model_names=model_names
    )
    _write_artifacts(result, out_dir)
    return result


def _write_artifacts(result: SelectionBenchmarkResult, out_dir: Path) -> None:
    csv_path = out_dir / "joint_eig_selection_timings.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result.records[0]).keys()))
        writer.writeheader()
        for rec in result.records:
            writer.writerow(asdict(rec))

    summary = {
        "model_names": result.model_names,
        "n_select": result.records[0].n_select,
        "n_scenarios": result.records[0].n_scenarios,
        "selections": {str(n): sel for n, sel in result.selections.items()},
    }
    (out_dir / "joint_eig_selection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot(result, out_dir / "joint_eig_selection.png")
    print(f"Wrote {csv_path}, summary JSON, and plot to {out_dir}")


def _plot(result: SelectionBenchmarkResult, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for stage, color in STAGE_COLORS.items():
        recs = sorted(
            (r for r in result.records if r.stage == stage), key=lambda r: r.n_pool
        )
        if not recs:
            continue
        ns = [r.n_pool for r in recs]
        secs = [r.seconds for r in recs]
        ax.plot(ns, secs, "o-", color=color, markersize=8, linewidth=2, label=stage)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Candidate pool size")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(
        f"Joint-EIG selection cost — k={result.records[0].n_select}, "
        f"{result.records[0].n_models} models, "
        f"{result.records[0].n_scenarios} scenarios"
    )
    ax.grid(True, which="both", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@dataclass
class Args:
    """Benchmark greedy joint-EIG stimulus-set selection at scale."""

    models_dir: Path = (
        REPO_ROOT / "src/pipelines/outer_loop/projects/subjective_randomness/seed_models"
    )
    """cognitive_models directory (models_manifest.yaml + PyMC .py files)."""
    out_dir: Path = REPO_ROOT / "data/analysis/joint_eig_selection"
    """Where the CSV / JSON / plot artifacts are written."""
    registry: Optional[Path] = None
    """Optional model_registry.yaml for a weighted model prior (uniform if omitted)."""
    pool_sizes: Tuple[int, ...] = (2000, 10000, 40000)
    """Sampled candidate-pool sizes to benchmark. Sampled pools are same-length
    pairs only (43,400 exist for lengths 4-8); the exhaustive pool adds
    cross-length pairs (122,760)."""
    exhaustive: bool = True
    """Also benchmark the full pair universe over the chosen lengths."""
    lengths: Tuple[int, ...] = (4, 5, 6, 7, 8)
    """Sequence lengths defining the pools and the universe."""
    n_select: int = 32
    """Stimuli to select per pool (the design stage's k)."""
    n_scenarios: int = 1000
    """Monte Carlo scenarios for gain estimation during selection."""
    n_samples: int = 200
    """Prior-predictive draws per model (pipeline default: 200)."""
    seed: int = 42
    """Seed for pools, prior draws, and scenarios."""
    n_random_baseline: int = 20
    """Random k-sets per pool for the baseline joint-EIG comparison."""


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
        pool_sizes=args.pool_sizes,
        exhaustive=args.exhaustive,
        lengths=args.lengths,
        n_select=args.n_select,
        n_scenarios=args.n_scenarios,
        n_samples=args.n_samples,
        seed=args.seed,
        out_dir=args.out_dir,
        model_weights=model_weights,
        n_random_baseline=args.n_random_baseline,
    )

    print("\n--- Joint-EIG selection at scale ---")
    largest = max(n for n, _ in ((r.n_pool, r) for r in result.records))
    for stage in ("featurize", "score", "select_exact", "select_lazy"):
        recs = [r for r in result.records if r.stage == stage and r.n_pool == largest]
        if recs:
            print(f"  {stage:>12s} @ {largest:,d} candidates: {recs[0].seconds:.1f}s")
    sel = result.selections[largest]
    print(
        f"  fresh joint EIG of the k={result.records[0].n_select} set: "
        f"exact={sel['exact_joint_eig_bits']:.4f}, "
        f"lazy={sel['lazy_joint_eig_bits']:.4f}, "
        f"random={sel['random_joint_eig_bits']['mean']:.4f} bits "
        f"(H(M) = {np.log2(len(result.model_names)):.3f} max)"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
