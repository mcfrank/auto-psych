"""Validation for 4_collect outputs."""

from pathlib import Path
from typing import Any, Dict

from src.validation.types import Validated


def validate_collect_output(run_dir: Path) -> Validated:
    run_dir = Path(run_dir)
    path = run_dir / "4_collect" / "responses.csv"
    details: Dict[str, Any] = {}

    if not path.exists():
        return Validated(False, "responses.csv not found", {"path": str(path)})

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) < 2:
        return Validated(False, "responses.csv has no data rows", details)
    header = lines[0].split(",")
    required = {"participant_id", "trial_index", "sequence_a", "sequence_b", "chose_left"}
    missing = required - set(h.strip() for h in header)
    if missing:
        return Validated(False, f"responses.csv missing columns: {missing}", details)

    details["n_rows"] = len(lines) - 1
    return Validated(True, f"Collect output valid; {len(lines)-1} rows", details)
