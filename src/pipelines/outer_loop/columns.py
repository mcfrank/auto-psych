"""Raw response rows and shared CSV writer for the pipeline.

Every model computes its own features from raw stimulus rows via its
``compute_features`` or ``prepare_observed`` hook. The pipeline passes only
the task's raw columns (``TaskSpec.raw_columns``, ``src.tasks``).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from src.tasks import TaskSpec


def raw_response_rows(rows: Iterable[Mapping[str, Any]], task: TaskSpec) -> List[Dict[str, Any]]:
    """Copy ``rows`` keeping only the task's raw columns, in that order.

    Everything agents can read carries only these. Collection returns more:
    ``/results`` adds ``participant_id_str`` (the Prolific ID), ``chose_right``
    and ``model``, and the simulated collectors ``chose_right`` and ``model``
    (the generating model). Fails loudly if a row lacks a raw column.
    """
    raw_rows = []
    for index, row in enumerate(rows):
        missing = [column for column in task.raw_columns if column not in row]
        if missing:
            raise ValueError(
                f"response row {index} lacks raw column(s) {missing}; got {sorted(row)}"
            )
        raw_rows.append({column: row[column] for column in task.raw_columns})
    return raw_rows


def write_responses_csv(rows: Sequence[Mapping[str, Any]], out_path: Path) -> Path:
    """Write response rows to a CSV. Returns the path written."""
    if not rows:
        raise ValueError("No response rows to write.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    # Through a temporary file: a truncated CSV with a header and one row
    # passes the collect validator, so a resume would skip the stage.
    partial = out_path.with_name(f".{out_path.name}.partial")
    with partial.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, out_path)
    return out_path
