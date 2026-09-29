"""The task description every inner-loop agent is told.

The outer loop copies the project's ``task_description.md`` next to the inner
loop's ``responses.csv`` (``model_loop_runner.write_task_description``); the
candidate and critique contexts inline it. Before, those agents saw the data's
columns but were never told what a trial is or what ``chose_left`` means.
"""

from __future__ import annotations

from pathlib import Path

TASK_DESCRIPTION_NAME = "task_description.md"


def read_task_description(responses_path: Path) -> str:
    """The task description beside the inner loop's ``responses.csv``."""
    path = Path(responses_path).parent / TASK_DESCRIPTION_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: inner-loop agents are told the task from it, and "
            "the outer loop copies it there from the project's assets "
            "(model_loop_runner.write_task_description)."
        )
    return path.read_text(encoding="utf-8").strip()
