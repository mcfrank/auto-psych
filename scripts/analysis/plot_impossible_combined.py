"""CLI: combine many impossible-ground-truth recovery runs into one figure.

The impossible test-retest sweep has the same layout as the holdout one, one
``holdout.json`` per (run, ground truth)::

    <runs_root>/run<r>/<gt_model>/holdout.json

but the ground truths are models *outside* the seed hypothesis space (e.g.
``more_heads_more_random``) rather than the seed models themselves. Each file is
a single run's recovery trajectory for one held-out ground truth (the same data
behind that directory's ``holdout.png``). This script pools the matching ground
truths across all runs and draws the same per-model panels as
``plot_holdout_combined.py`` — the best-model recovery trajectory as a mean with
per-step error bars, and the seed-model baselines as means with a shaded spread
band — then writes a tidy CSV of every pooled point.

It reuses the holdout aggregation, which leaves a cell out of a point — loop
and baselines alike — where a value is degenerate (``None``, ``NaN``, or
``inf``: common for impossible ground truths, where a constant prediction
makes the correlation undefined) instead of crashing, and lists it.

It fails loudly if no run files are found under ``--runs-root``.

Points are aligned by experiment (its seed step, the rounds every cell ran, its
end), and at each point the loop and every baseline cover the same cells (see
``aggregate_holdout_trajectories``). ``<stem>_cells.md`` lists the expected
cells that are partial or missing and, per point, the complete cells left out
and why. Pass ``--n-repeats``/``--gt-models`` so cells that never started are
listed too.

Usage:
    # Default: pool data/results/impossible_holdout_test_retest, both metrics
    uv run python scripts/analysis/plot_impossible_combined.py

    # Explicit source, just the RMSE figure, std error bars
    uv run python scripts/analysis/plot_impossible_combined.py \\
        --runs-root data/results/impossible_holdout_test_retest \\
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
    AGGREGATE_TIDY_COLUMNS,
    DEFAULT_TRAJECTORY_X_LABEL,
    aggregate_exclusion_lines,
    aggregate_holdout_trajectories,
    aggregate_tidy_rows,
    plot_holdout_trajectories_combined,
)
from src.subjective_randomness.sweep_cells import (  # noqa: E402
    accounting_lines,
    survey_sweep,
)
from src.subjective_randomness.tidy import write_tidy_csv  # noqa: E402

DEFAULT_RUNS_ROOT = Path("data/results/impossible_holdout_test_retest")

# Impossible ground truths lie outside the seed family, so no seed is held out:
# the fitted-seed baseline is just the best seed model (not the best *other* seed
# model as in holdout recovery, where the ground truth is itself a seed).
FITTED_BASELINE_LABEL = "best seed model"

# Impossible ground-truth names are long (e.g. "more_imbalance_more_random"), so
# the default panel-heading size overlaps adjacent facets. Shrink it for these
# figures; the shorter-named holdout-recovery figures keep the default.
STRIP_TEXT_SIZE = 15

# Both metrics the recovery figure understands, in the order figures are written.
ALL_METRICS = ("rmse", "pearson_r")

TIDY_COLUMNS = AGGREGATE_TIDY_COLUMNS


@dataclass
class Args:
    """Combine impossible-ground-truth recovery runs into mean ± error figures."""

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
    n_repeats: Optional[int] = None
    """The sweep's repeat count, so cells that never started are listed as
    missing (default: inferred from the run<r>/ directories, and said so)."""
    gt_models: Optional[str] = None
    """The sweep's ground truths, space-separated (as GT_MODELS; default:
    inferred from the directories present, and said so)."""
    name_suffix: str = ""
    """Appended to the output filename stem to mark a variant, e.g.
    ``--name-suffix _no_inner_loop`` -> ``impossible_combined_no_inner_loop_rmse.pdf``."""


def find_run_files(runs_root: Path) -> List[Path]:
    """All per-(run, ground-truth) ``holdout.json`` files under ``runs_root``."""
    return sorted(runs_root.glob("run*/*/holdout.json"))


def load_results(files: Iterable[Path]) -> List[Mapping[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def tidy_rows(aggregated: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """One tidy row per plotted point (``aggregate_tidy_rows``)."""
    return aggregate_tidy_rows(aggregated)


def main(args: Args) -> None:
    runs_root = resolve_path(args.runs_root)
    out_dir = resolve_path(args.out_dir) if args.out_dir is not None else runs_root

    files = find_run_files(runs_root)
    if not files:
        raise FileNotFoundError(
            f"No run files matched {runs_root}/run*/*/holdout.json — nothing to pool."
        )
    # Every expected cell, so the unfinished ones are listed, not silently absent.
    survey = survey_sweep(
        runs_root,
        n_repeats=args.n_repeats,
        gt_models=args.gt_models.split() if args.gt_models else None,
    )
    files = [cell / "holdout.json" for _, cell in sorted(survey.complete.items())]
    labels = sorted(survey.complete)
    results = load_results(files)
    print(f"Pooling {len(files)} impossible-ground-truth run file(s) under {runs_root}")

    metrics = ALL_METRICS if args.metric == "both" else (args.metric,)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One CSV holds every metric's pooled points; figures are per metric.
    stem = f"impossible_combined{args.name_suffix}"
    all_rows: List[Mapping[str, Any]] = []
    cell_lines = accounting_lines(survey, included=labels)
    for metric in metrics:
        aggregated = aggregate_holdout_trajectories(
            results, metric=metric, error=args.error, labels=labels
        )
        figure_path = out_dir / f"{stem}_{metric}.pdf"
        plot_holdout_trajectories_combined(
            aggregated,
            figure_path,
            fitted_baseline_label=FITTED_BASELINE_LABEL,
            x_label=args.x_label,
            strip_text_size=STRIP_TEXT_SIZE,
        )
        print(f"Wrote {metric} figure to {figure_path}")
        all_rows.extend(tidy_rows(aggregated))
        cell_lines.extend(aggregate_exclusion_lines(aggregated))

    csv_path = out_dir / f"{stem}.csv"
    write_tidy_csv(all_rows, csv_path, columns=TIDY_COLUMNS)
    print(f"Wrote tidy pooled CSV to {csv_path}")
    cells_path = out_dir / f"{stem}_cells.md"
    cells_path.write_text("\n".join(cell_lines) + "\n", encoding="utf-8")
    print("\n".join(accounting_lines(survey, included=labels)))
    print(f"Wrote the cells each point covers to {cells_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
