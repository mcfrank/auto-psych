"""Redo the rounds a finished experiment abandoned, from ``--from-round`` on.

For a holdout cell whose model loop lost rounds (2026-10-30: a usage limit the
harness did not recognise) and then ended the experiment as if they had run.
The cell's run tree is first copied to ``<cell>/superseded/<time>/`` (outside
the agent tree), later experiments are moved there, and the experiment's loop
is checkpointed before the round (``checkpoint_before_round``); resuming the
cell then continues the loop from that round.

    python scripts/subjective_randomness/redo_abandoned_rounds.py \\
        --cell <WORK_ROOT>/run2/<gt> --experiment 1 --from-round 1
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.pipelines.inner_loop.round_checkpoint import checkpoint_before_round  # noqa: E402
from src.pipelines.outer_loop.model_loop_runner import (  # noqa: E402
    AGENT_NOTES_SNAPSHOT_DIRNAME,
    agent_notes_dir,
)
from src.pipelines.outer_loop.orchestrator_validators import EXPORT_RECORD_FILENAME  # noqa: E402
from src.runtime.atomic_files import copy_directory_atomically  # noqa: E402


@dataclass
class Args:
    cell: Path
    """The holdout cell directory (holds ``repo/``)."""
    experiment: int
    from_round: int
    """The first abandoned round."""


def main(args: Args) -> None:
    (run_root,) = [p for p in (args.cell / "repo" / "_runs").iterdir() if p.is_dir()]
    exp = run_root / f"experiment{args.experiment}"
    later = [p for p in run_root.glob("experiment*") if int(p.name[len("experiment"):]) > args.experiment]
    superseded = args.cell / "superseded" / time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(run_root, superseded / "before_repair")
    # The notes as the experiment ended: the next experiment's snapshot of them.
    snapshots = [p / AGENT_NOTES_SNAPSHOT_DIRNAME for p in sorted(later) if (p / AGENT_NOTES_SNAPSHOT_DIRNAME).is_dir()]
    if snapshots:
        copy_directory_atomically(snapshots[0], agent_notes_dir(exp))
    checkpoint_before_round(exp / "model_loop", args.from_round, agent_notes_dir(exp), ledger_context=exp.name)
    for p in later:
        shutil.rmtree(p)
    (exp / "model_loop" / EXPORT_RECORD_FILENAME).unlink()
    print(f"[redo] {exp}: resumes from round {args.from_round}; prior state in {superseded}")


if __name__ == "__main__":
    main(tyro.cli(Args))
