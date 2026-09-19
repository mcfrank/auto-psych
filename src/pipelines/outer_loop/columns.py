"""Raw response columns and shared CSV writer for the pipeline.

Every model computes its own features from raw stimulus rows via its
``compute_features`` or ``prepare_observed`` hook. The pipeline passes only
the raw columns below.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

RAW_RESPONSE_COLUMNS = (
    "sequence_a",
    "sequence_b",
    "participant_id",
    "trial_index",
    "chose_left",
)


def write_responses_csv(
    rows: Sequence[Mapping[str, Any]], out_path: Path
) -> Path:
    """Write response rows to a CSV. Returns the path written."""
    if not rows:
        raise ValueError("No response rows to write.")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path
