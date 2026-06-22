#!/usr/bin/env python3
"""CLI: move a finished live (human) outer-loop run's results into the repo.

Copies the *real* runs' experiment trees (responses, design, cognitive models,
model-loop results, agent logs) from the scratch ``WORK_ROOT`` into
``data/results/human_experiment``, while:

* excluding heavy non-results (per-run repo worktrees, node_modules, language
  caches, and the multi-100MB MCMC ``.nc`` fit-caches), and
* scrubbing EVERY Prolific id — the raw worker id (``participant_id_str`` column
  of responses.csv) and the bare 24-hex worker/study ids that also appear in
  logs, deployment manifests, configs and agent transcripts.

Pilots and ``_validate*`` runs are skipped by default. Collection fails loudly
if any Prolific id would survive into the repo. A generated ``SUMMARY.md``
reports the winning model, its posterior, and the ELPD margin per (run,
experiment).

Usage:
    # Default: real runs on $SCRATCH -> data/results/human_experiment
    uv run python scripts/outer_loop_live/collect_human_results.py

    uv run python scripts/outer_loop_live/collect_human_results.py \\
        --source $SCRATCH/auto-psych/outer_loop_live \\
        --dest data/results/human_experiment \\
        --runs run1 run2 run3
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

from src.pipelines.outer_loop.results_collection import (  # noqa: E402
    collect_human_results,
)

DEFAULT_SOURCE_RELATIVE = Path("auto-psych/outer_loop_live")
DEFAULT_DEST_RELATIVE = Path("data/results/human_experiment")


@dataclass
class Args:
    """Move a finished live (human) outer-loop run's results into the repo."""

    source: Optional[Path] = None
    """Scratch WORK_ROOT holding <run>/data/<project>/experiment*
    (default: $SCRATCH/auto-psych/outer_loop_live)."""
    dest: Path = DEFAULT_DEST_RELATIVE
    """Destination inside the repo (relative paths resolve against the repo root)."""
    project: str = "subjective_randomness"
    """Project whose experiment trees to collect."""
    runs: Optional[list[str]] = None
    """Explicit run labels to collect (default: auto-discover real runs)."""
    include_pilots: bool = False
    """Also collect pilot* runs (default: real runs only)."""
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

    report = collect_human_results(
        source,
        dest,
        project=args.project,
        runs=args.runs,
        include_pilots=args.include_pilots,
        overwrite=args.overwrite,
    )

    print(f"Collected live (human) results from {report.source}")
    print(f"  -> {report.dest}")
    print(f"  runs: {', '.join(report.runs)}")
    print(f"  experiments copied: {report.n_experiments}")
    print(f"  Prolific ids scrubbed: {report.n_ids_scrubbed}")
    print(f"  wrote summary: {report.summary_path.relative_to(report.dest)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
