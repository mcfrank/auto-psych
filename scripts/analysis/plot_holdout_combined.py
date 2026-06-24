"""CLI: combine many holdout-recovery runs into one figure with error bars.

A holdout test-retest sweep leaves one ``holdout.json`` per (run, ground truth)::

    <runs_root>/run<r>/<gt_model>/holdout.json

Each file is a single run's recovery trajectory for one held-out ground truth
(the same data behind that directory's ``holdout.png``). This script pools the
matching ground truths across all runs and draws the same per-model panels (with
plotnine), showing the best-model recovery trajectory as a mean with per-step
error bars across runs and the seed-model baselines as means with a shaded spread
band. It also writes a tidy CSV of every pooled point.

It fails loudly if no run files are found under ``--runs-root``.

Usage:
    # Default: pool data/results/holdout_test_retest, write both metrics' figures
    uv run python scripts/analysis/plot_holdout_combined.py

    # Explicit source, just the RMSE figure, std error bars
    uv run python scripts/analysis/plot_holdout_combined.py \\
        --runs-root data/results/holdout_test_retest \\
        --metric rmse --error std
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402
from src.subjective_randomness.reporting import (  # noqa: E402
    DEFAULT_TRAJECTORY_X_LABEL,
    aggregate_holdout_trajectories,
    plot_holdout_trajectories_combined,
)
from src.subjective_randomness.tidy import write_tidy_csv  # noqa: E402

DEFAULT_RUNS_ROOT = Path("data/results/holdout_test_retest")

# Both metrics the holdout figure understands, in the order figures are written.
ALL_METRICS = ("rmse", "pearson_r")

TIDY_COLUMNS = [
    "gt_model",
    "metric",
    "error",
    "series",
    "global_step",
    "mean",
    "err",
    "n",
]


@dataclass
class Args:
    """Combine holdout-recovery runs into mean ± error figures and a tidy CSV."""

    runs_root: Path = DEFAULT_RUNS_ROOT
    """Directory holding ``run<r>/<gt_model>/holdout.json`` trees to pool."""
    out_dir: Optional[Path] = None
    """Where to write figures and the CSV (default: ``--runs-root``)."""
    metric: Literal["rmse", "pearson_r", "both"] = "both"
    """Which figure(s) to draw: one metric, or ``both``."""
    error: Literal["sem", "std", "ci95"] = "sem"
    """Spread shown by the error bars/band: standard error, standard deviation,
    or a 95% normal interval half-width."""
    x_label: str = DEFAULT_TRAJECTORY_X_LABEL
    """X-axis label. The default fits the full pipeline; the no-inner-loop ablation
    has one step per experiment, so pass ``--x-label experiment`` for it."""
    name_suffix: str = ""
    """Appended to the output filename stem to mark a variant, e.g.
    ``--name-suffix _no_inner_loop`` -> ``holdout_combined_no_inner_loop_rmse.pdf``."""


def find_run_files(runs_root: Path) -> List[Path]:
    """All per-(run, ground-truth) ``holdout.json`` files under ``runs_root``."""
    return sorted(runs_root.glob("run*/*/holdout.json"))


def load_results(files: Iterable[Path]) -> List[Mapping[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def tidy_rows(aggregated: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Flatten a pooled aggregate into one tidy row per plotted point.

    Trajectory points carry their ``global_step``; the two flat seed-model
    baselines (constant across steps) carry an empty ``global_step``.
    """
    metric, error = aggregated["metric"], aggregated["error"]
    rows: List[Mapping[str, Any]] = []
    for panel in aggregated["gt_models"]:
        gt = panel["gt_model"]
        for point in panel["best"]:
            rows.append(
                {
                    "gt_model": gt,
                    "metric": metric,
                    "error": error,
                    "series": "best",
                    "global_step": point["global_step"],
                    "mean": point["mean"],
                    "err": point["err"],
                    "n": point["n"],
                }
            )
        for series, stats in panel["baselines"].items():
            if stats is not None:
                rows.append(
                    {
                        "gt_model": gt,
                        "metric": metric,
                        "error": error,
                        "series": series,
                        "global_step": "",
                        "mean": stats["mean"],
                        "err": stats["err"],
                        "n": stats["n"],
                    }
                )
    return rows


def main(args: Args) -> None:
    runs_root = resolve_path(args.runs_root)
    out_dir = resolve_path(args.out_dir) if args.out_dir is not None else runs_root

    files = find_run_files(runs_root)
    if not files:
        raise FileNotFoundError(
            f"No run files matched {runs_root}/run*/*/holdout.json — nothing to pool."
        )
    results = load_results(files)
    print(f"Pooling {len(files)} run file(s) under {runs_root}")

    metrics = ALL_METRICS if args.metric == "both" else (args.metric,)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One CSV holds every metric's pooled points; figures are per metric.
    stem = f"holdout_combined{args.name_suffix}"
    all_rows: List[Mapping[str, Any]] = []
    for metric in metrics:
        aggregated = aggregate_holdout_trajectories(
            results, metric=metric, error=args.error
        )
        figure_path = out_dir / f"{stem}_{metric}.pdf"
        plot_holdout_trajectories_combined(
            aggregated, figure_path, x_label=args.x_label
        )
        print(f"Wrote {metric} figure to {figure_path}")
        all_rows.extend(tidy_rows(aggregated))

    csv_path = out_dir / f"{stem}.csv"
    write_tidy_csv(all_rows, csv_path, columns=TIDY_COLUMNS)
    print(f"Wrote tidy pooled CSV to {csv_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
