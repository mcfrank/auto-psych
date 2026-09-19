"""CLI: compute the recovery ceiling for every cell of a finished sweep.

The ceiling is the held-out ground-truth model refit on the cell's own training
responses and scored on the cell's held-out pool. It is the best RMSE any model
with the right functional form could reach on this data, so it separates what
the loop failed to *find* from what the data cannot *support*. See
``src/subjective_randomness/recovery_ceiling.py`` for the argument.

No agents and no new sampling beyond the ground-truth refits themselves; the
cells' own ``mcmc_cache`` is reused, so a re-run is nearly free.

Usage:
    uv run python scripts/subjective_randomness/recovery_ceiling.py \\
        --sweep $SCRATCH/auto-psych/consolidation_2026_09/sweep_rerun \\
        --out   $SCRATCH/auto-psych/consolidation_2026_09/ceiling/sweep_rerun
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.recovery_ceiling import (  # noqa: E402
    run_ceiling,
    summarize_by_ground_truth,
)


@dataclass
class Args:
    """Score every cell of a sweep against its achievable ceiling."""

    sweep: Path
    """A finished sweep root (holds run<r>/<gt>/holdout.json)."""
    out: Path
    """Output directory; ceiling.csv, ceiling.json and summary.md are written here."""
    gt_models_dir: Optional[Path] = None
    """Registry holding the held-out models. Default: each cell's recorded
    seed_models_dir, which points into the sweep's staged harness repo."""
    predict_max_draws: Optional[int] = 500
    """Thin the posterior when predicting over the exhaustive pool."""
    draws: Optional[int] = None
    """Override the cell's recorded sampler settings (default: reuse them)."""
    tune: Optional[int] = None
    chains: Optional[int] = None
    limit_eval: Optional[int] = None
    """Score only the first N eval stimuli. For smoke-testing this CLI only —
    a real run must leave it unset."""


def main(args: Args) -> None:
    fit_kwargs = {
        k: v
        for k, v in (("draws", args.draws), ("tune", args.tune), ("chains", args.chains))
        if v is not None
    }
    print(f"[ceiling] sweep {args.sweep}")
    if args.limit_eval is not None:
        print(f"[ceiling] WARNING: limit_eval={args.limit_eval} — smoke mode, not a real result")

    run = run_ceiling(
        args.sweep,
        fit_kwargs=fit_kwargs or None,
        predict_max_draws=args.predict_max_draws,
        limit_eval=args.limit_eval,
        gt_models_dir=args.gt_models_dir,
    )
    summary = summarize_by_ground_truth(run)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = [c.as_row() for c in run.cells]
    if rows:
        with (args.out / "ceiling.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.out / "ceiling.json").write_text(
        json.dumps(
            {
                "sweep": str(args.sweep),
                "limit_eval": args.limit_eval,
                "cells": rows,
                "unscored": run.unscored,
                "by_ground_truth": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# Recovery ceiling — {args.sweep.name}",
        "",
        "The ceiling is the held-out ground-truth model refit on each cell's own",
        "training data and scored on its held-out pool: the best RMSE the right",
        "functional form can reach here. `gap` is the loop's RMSE minus the ceiling —",
        "the part attributable to search rather than to finite data.",
        "",
        f"Cells scored: {len(run.cells)}; unscored: {len(run.unscored)}",
        "",
        "| ground truth | n | mean ceiling | max ceiling | mean loop RMSE | mean gap |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary:
        def fmt(v):
            return "n/a" if v is None else f"{v:.4f}"
        lines.append(
            f"| {row['gt_model']} | {row['n']} | {fmt(row['mean_ceiling_rmse'])} | "
            f"{fmt(row['max_ceiling_rmse'])} | {fmt(row['mean_loop_rmse'])} | {fmt(row['mean_gap'])} |"
        )
    if run.unscored:
        lines += ["", "## Unscored cells", ""]
        lines += [f"- `{u['cell']}`: {u['reason']}" for u in run.unscored]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n[ceiling] wrote {args.out}/ceiling.csv, ceiling.json, summary.md")


if __name__ == "__main__":
    main(tyro.cli(Args))
