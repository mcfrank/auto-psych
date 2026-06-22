#!/usr/bin/env python3
"""CLI: move a finished holdout test-retest run's results into the repo.

A holdout test-retest run leaves its output under a scratch ``WORK_ROOT``::

    test_retest.{json,csv,png}            aggregate reliability across repeats
    run<r>/<gt>/holdout.{json,csv,png}    per-(repeat, ground-truth) trajectory

alongside heavy material we never commit (per-task repo copies, MCMC ``.nc``
caches, the shared venv, agent state). This copies only the lightweight result
artifacts into ``data/results/holdout_test_retest`` and writes a generated
``SUMMARY.md`` distilled from the aggregate JSON. It fails loudly if expected
artifacts are missing or the destination is already populated.

Usage:
    # Default: most recent run on $SCRATCH -> data/results/holdout_test_retest
    uv run python scripts/subjective_randomness/collect_holdout_results.py

    # Explicit source / destination, or just the aggregate summary:
    uv run python scripts/subjective_randomness/collect_holdout_results.py \\
        --source $SCRATCH/auto-psych/holdout_test_retest_full \\
        --dest data/results/holdout_test_retest \\
        --summary-only
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.results_collection import collect_results  # noqa: E402

# The default run is the most recent test-retest loop on scratch; override with
# --source for any other run.
DEFAULT_SOURCE_RELATIVE = Path("auto-psych/holdout_test_retest_full")
DEFAULT_DEST_RELATIVE = Path("data/results/holdout_test_retest")


@dataclass
class Args:
    """Move a finished holdout test-retest run's results into the repo."""

    source: Optional[Path] = None
    """Scratch run root holding test_retest.* and run<r>/<gt>/holdout.*
    (default: $SCRATCH/auto-psych/holdout_test_retest_full)."""
    dest: Path = DEFAULT_DEST_RELATIVE
    """Destination inside the repo (relative paths resolve against the repo root)."""
    summary_only: bool = False
    """Copy only the aggregate test_retest.* and SUMMARY.md, skipping the
    per-run holdout.* artifacts."""
    overwrite: bool = False
    """Replace the destination's contents if it already exists and is non-empty."""


def _resolve_source(source: Optional[Path]) -> Path:
    if source is not None:
        return source.expanduser()
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise SystemExit(
            "No --source given and $SCRATCH is unset; pass --source explicitly."
        )
    return Path(scratch) / DEFAULT_SOURCE_RELATIVE


def main(args: Args) -> None:
    source = _resolve_source(args.source)
    dest = args.dest if args.dest.is_absolute() else here() / args.dest

    report = collect_results(
        source,
        dest,
        include_per_run=not args.summary_only,
        overwrite=args.overwrite,
    )

    print(f"Collected holdout test-retest results from {report.source}")
    print(f"  -> {report.dest}")
    print(f"  aggregate files: {', '.join(p.name for p in report.aggregate_paths)}")
    print(f"  per-run artifacts copied: {report.n_per_run_artifacts}")
    print(f"  wrote summary: {report.summary_path.relative_to(report.dest)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
