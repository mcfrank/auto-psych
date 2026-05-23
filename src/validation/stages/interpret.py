"""Validation for 6_interpret outputs."""

from pathlib import Path
from typing import Any, Dict

from src.validation.types import Validated


def validate_interpreter_output(run_dir: Path) -> Validated:
    run_dir = Path(run_dir)
    path = run_dir / "6_interpret" / "report.md"
    details: Dict[str, Any] = {}

    if not path.exists():
        return Validated(False, "report.md not found", {"path": str(path)})
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return Validated(False, "report.md is empty", details)
    details["length"] = len(text)
    return Validated(True, "Interpreter output valid", details)
