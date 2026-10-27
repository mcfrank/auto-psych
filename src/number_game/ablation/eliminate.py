"""Backward elimination over the number-game final's mechanisms, by CV grouped by example set.

    python -m src.number_game.ablation.eliminate --final ng --data <responses.csv> --out <dir> --mode plan
    python -m src.number_game.ablation.eliminate --final ng --data <responses.csv> --out <dir> --mode worker --task <i>

The rule, ``plan`` and ``worker`` are `src.ablation.eliminate`'s, unchanged
from the sr-oct26 ablation (docs/ng_ablation.md, fixed before any fit). What
is the number game's: the final (`src.number_game.ablation.final`), the data
(the run's own 7,560 responses, `src.number_game.ablation.data`), the CV unit
and the SE's clusters (the example set: ``number_set``, the unit the run
itself selected by), and that a worker array task fits ``tasks_per_job``
(model, fold) tasks in turn, since one fit takes only ~6 minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Literal

import numpy as np
import tyro

from src.ablation import eliminate as E
from src.number_game.ablation import variants as V

CV_GROUP = ("number_set",)


def example_set_clusters(rows) -> np.ndarray:
    """One integer per row; rows of one example set share it (numbered in order of first appearance)."""
    from src.model_comparison.cv import cv_units

    ids: dict = {}
    return np.array([ids.setdefault(unit, len(ids)) for unit in cv_units(rows, CV_GROUP)], dtype=np.int64)


ABLATION = E.Ablation(variants=V.VARIANTS, cv_group=CV_GROUP, units=example_set_clusters)


@dataclass
class Args(E.Args):
    final: Literal["ng"]
    data: Path
    """The run's cumulative responses (`src.number_game.ablation.data`); the CV folds are written beside them."""
    time_limit_hours: float = 1.0
    """worker: per sampling attempt; a timed-out attempt moves on to the next seed."""
    tasks_per_job: int = 5
    """worker: array task ``task`` fits ``next.json``'s tasks ``task * tasks_per_job`` onwards, this many."""


def plan(args: Args) -> dict:
    """Replay the rule over the saved fold results: the next round's missing fits, or the result."""
    return E.plan(ABLATION, args)


def worker(args: Args) -> List[dict]:
    """Fit the array task's ``tasks_per_job`` tasks of ``next.json`` in turn."""
    import json

    n_tasks = len(json.loads((Path(args.out) / "next.json").read_text()).get("tasks", []))
    first = args.task * args.tasks_per_job
    if not 0 <= first < n_tasks:
        raise ValueError(f"array task {args.task} has no tasks: next.json has {n_tasks}, {args.tasks_per_job} per job")
    return [E.worker(ABLATION, replace(args, task=t)) for t in range(first, min(first + args.tasks_per_job, n_tasks))]


def main(args: Args):
    return plan(args) if args.mode == "plan" else worker(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
