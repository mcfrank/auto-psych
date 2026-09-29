"""Record a checkout's git commit into a run copy made without ``.git``.

The live launchers (``run_pilot.sh``, ``submit_parallel.sh``) run each job
from an rsync copy of the checkout that leaves ``.git`` out; the deploy
records the commit its code came from and raises without one. The launchers
run this right after the copy:

    python -m src.pipelines.outer_loop.deployment.record_provenance \\
        --checkout "$REPO" --copy "$WT"

If you rsync fixed code into a run copy by hand, run it again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro

from src.pipelines.outer_loop.deployment.manifest import record_code_provenance


@dataclass
class Args:
    """Record a checkout's git commit (and dirtiness) into its run copy."""

    checkout: Path
    """The git checkout the copy was made from."""
    copy: Path
    """The run copy (no .git) the job will deploy from."""


def main(args: Args) -> None:
    record = record_code_provenance(args.checkout, args.copy)
    print(f"[provenance] recorded {args.checkout}'s commit in {record}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
