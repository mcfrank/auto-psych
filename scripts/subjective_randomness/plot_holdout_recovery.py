"""CLI: re-analyze and plot a finished holdout-recovery run.

Reads an existing holdout-recovery result JSON (the combined ``--out`` of
``holdout_recovery.py``) and, for every held-out ground truth, recomputes both
trajectories from its on-disk run tree through the shared MCMC cache:

* the single best-fitting model's correlation with the ground truth, and
* the posterior-weighted Bayesian model average's correlation,

at every inner-loop scoring step. It then rewrites the enriched JSON, a tidy
CSV, and the two-line figure. Because the per-step refits are cache hits, this
is fast — it never re-runs the (hours-long) agentic loop or resamples MCMC.

Usage:
    uv run python scripts/subjective_randomness/plot_holdout_recovery.py \\
        --result data/subjective_randomness/holdout_recovery/holdout.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.config import resolve_path  # noqa: E402
from src.subjective_randomness.holdout_recovery import (  # noqa: E402
    TRAJECTORY_COLUMNS,
    reevaluate_trajectories,
    trajectory_tidy_rows,
)
from src.subjective_randomness.reporting import plot_holdout_trajectories  # noqa: E402
from src.subjective_randomness.tidy import write_tidy_csv  # noqa: E402


@dataclass
class Args:
    """Re-analyze and plot a finished holdout-recovery result."""

    result: Path
    """Existing holdout-recovery result JSON to re-analyze."""
    figure: Optional[Path] = None
    """Output figure path (default: <result dir>/<result stem>.png)."""
    tidy_csv: Optional[Path] = None
    """Output tidy CSV path (default: <result dir>/<result stem>.csv)."""
    out: Optional[Path] = None
    """Where to write the enriched result JSON (default: overwrite --result)."""
    cache_dir: Optional[Path] = None
    """Shared PyMC .nc fit cache (default: <result dir>/mcmc_cache). The cache
    is what makes the per-step refits free."""
    seed_models_dir: Optional[Path] = None
    """Seed-model directory (default: the result's recorded seed_models_dir)."""


def main(args: Args) -> None:
    result_path = resolve_path(args.result)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    cache_dir = (
        resolve_path(args.cache_dir)
        if args.cache_dir is not None
        else result_path.parent / "mcmc_cache"
    )
    seed_models_dir = (
        resolve_path(args.seed_models_dir)
        if args.seed_models_dir is not None
        else resolve_path(result["seed_models_dir"])
    )

    enriched = reevaluate_trajectories(
        result, seed_models_dir=seed_models_dir, cache_dir=cache_dir
    )

    out_path = resolve_path(args.out) if args.out is not None else result_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    print(f"Wrote enriched holdout-recovery result to {out_path}")
    def _fmt(value: object) -> str:
        return "undefined" if value is None else f"{value:.3f}"

    for gt_run in enriched["gt_runs"]:
        final = gt_run["trajectory"][-1]
        print(
            f"  {gt_run['gt_model']}: final best r={_fmt(final['pearson_r'])} "
            f"(best={final['best_model']}), BMA r={_fmt(final['pearson_r_bma'])}, "
            f"fitted-seed r={_fmt(gt_run['fitted_baseline']['mean_r'])}, "
            f"default-seed r={_fmt(gt_run['baseline']['mean_r'])}"
        )

    tidy_path = (
        resolve_path(args.tidy_csv)
        if args.tidy_csv is not None
        else result_path.with_suffix(".csv")
    )
    write_tidy_csv(trajectory_tidy_rows(enriched), tidy_path, columns=TRAJECTORY_COLUMNS)
    print(f"Wrote tidy trajectory CSV to {tidy_path}")

    figure_path = (
        resolve_path(args.figure)
        if args.figure is not None
        else result_path.with_suffix(".png")
    )
    plot_holdout_trajectories(enriched, figure_path, metric="pearson_r")
    print(f"Wrote trajectory figure (Pearson r) to {figure_path}")

    rmse_figure_path = figure_path.with_name(f"{figure_path.stem}_rmse{figure_path.suffix}")
    plot_holdout_trajectories(enriched, rmse_figure_path, metric="rmse")
    print(f"Wrote trajectory figure (RMSE) to {rmse_figure_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))
