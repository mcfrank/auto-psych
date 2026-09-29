"""Which holdout cells to resume after an array, and which are missing.

    # Tasks of a finished array to resume, from sacct:
    sacct -j <array_id> -X -n -P -o JobID,State \
      | python cell_status.py retry-plan --work-root W --gt-models "a b c d"
    # Expected cells with no holdout.json:
    python cell_status.py missing --work-root W --n-repeats 5 --gt-models "a b c d"

Cells used to die on a time limit or out of memory and then vanish from every
report (survivorship). The retry job resumes them; the summary job lists
whatever is still missing.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Union

import tyro

# Slurm end states worth another attempt: the cell resumes from its last
# finished stage. A cancelled task is the user's decision and is left alone.
RETRY_STATES = {"FAILED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
OUT_OF_MEMORY = "OUT_OF_MEMORY"


def task_cell(task: int, gt_models: Sequence[str]) -> str:
    """The ``run<r>/<gt>`` cell of array task ``task`` (as the array maps it)."""
    repeat = (task - 1) // len(gt_models) + 1
    return f"run{repeat}/{gt_models[(task - 1) % len(gt_models)]}"


def retry_plan(
    sacct_output: str, *, work_root: Path, gt_models: Sequence[str]
) -> Dict[str, List[int]]:
    """Task ids to resume, from ``sacct -X -n -P -o JobID,State`` output.

    ``more_memory`` holds the tasks that ran out of memory; ``same_memory``
    the other failures. A task whose cell already has its ``holdout.json`` is
    finished whatever Slurm says (a task that finished can be marked
    ``OUT_OF_MEMORY`` or time out in its final clean-up) and is not resumed.
    """
    plan: Dict[str, List[int]] = {"same_memory": [], "more_memory": []}
    for line in sacct_output.splitlines():
        if not line.strip():
            continue
        job_id, state = line.split("|", 1)
        if "_" not in job_id or "[" in job_id:
            continue  # not a single array task
        task = int(job_id.rsplit("_", 1)[1])
        if (Path(work_root) / task_cell(task, gt_models) / "holdout.json").exists():
            continue
        state = state.split()[0]
        if state == OUT_OF_MEMORY:
            plan["more_memory"].append(task)
        elif state in RETRY_STATES:
            plan["same_memory"].append(task)
    return {key: sorted(tasks) for key, tasks in plan.items()}


def missing_cells(
    work_root: Path, *, n_repeats: int, gt_models: Sequence[str]
) -> List[str]:
    """Every expected ``run<r>/<gt>`` cell that has no ``holdout.json``."""
    return [
        f"run{repeat}/{gt}"
        for repeat in range(1, n_repeats + 1)
        for gt in gt_models
        if not (Path(work_root) / f"run{repeat}" / gt / "holdout.json").exists()
    ]


@dataclass
class RetryPlan:
    """Read sacct output on stdin; print the plan as JSON."""

    work_root: Path
    gt_models: str
    """Space-separated, in array order (as GT_MODELS)."""


@dataclass
class Missing:
    """Print the expected cells with no result, one per line."""

    work_root: Path
    n_repeats: int
    gt_models: str
    """Space-separated, in array order (as GT_MODELS)."""


def main(command: Union[RetryPlan, Missing]) -> None:
    if isinstance(command, RetryPlan):
        print(
            json.dumps(
                retry_plan(
                    sys.stdin.read(),
                    work_root=command.work_root,
                    gt_models=command.gt_models.split(),
                )
            )
        )
    else:
        for cell in missing_cells(
            command.work_root,
            n_repeats=command.n_repeats,
            gt_models=command.gt_models.split(),
        ):
            print(cell)


if __name__ == "__main__":
    main(tyro.cli(Union[RetryPlan, Missing]))
