"""Validation for 5_analyze outputs."""

from pathlib import Path
from typing import Any, Dict

from src.validation.types import Validated


def validate_analyst_output(run_dir: Path) -> Validated:
    run_dir = Path(run_dir)
    agent_dir = run_dir / "5_analyze"
    agg_path = agent_dir / "aggregate.csv"
    summary_path = agent_dir / "summary_stats.json"
    details: Dict[str, Any] = {}

    if not agg_path.exists():
        return Validated(False, "aggregate.csv not found", {"path": str(agg_path)})
    lines = agg_path.read_text(encoding="utf-8").strip().split("\n")
    if not lines:
        return Validated(False, "aggregate.csv is empty", details)
    header = set(lines[0].split(","))
    if "chose_left_pct" not in header and "sequence_a" not in header:
        return Validated(
            False, "aggregate.csv missing expected columns", {"header": list(header)}
        )

    if not summary_path.exists():
        return Validated(
            False, "summary_stats.json not found", {"path": str(summary_path)}
        )
    try:
        data = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Validated(False, f"Invalid summary_stats.json: {exc}", details)
    if not isinstance(data, dict):
        return Validated(False, "summary_stats.json is not a dict", details)

    return Validated(True, "Data analyst output valid", details)
