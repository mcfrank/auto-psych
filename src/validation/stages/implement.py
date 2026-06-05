"""Validation for 3_implement outputs."""

from pathlib import Path
from typing import Any, Dict

from src.validation.types import Validated


def validate_implementer_output(run_dir: Path) -> Validated:
    run_dir = Path(run_dir)
    impl_dir = run_dir / "3_implement"
    index_path = impl_dir / "index.html"
    config_path = impl_dir / "config.json"
    details: Dict[str, Any] = {}

    if not config_path.exists():
        return Validated(
            False, "config.json not found (deploy step)", {"path": str(config_path)}
        )
    try:
        data = __import__("json").loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Validated(
            False, f"Invalid config.json: {exc}", {"path": str(config_path)}
        )
    if not isinstance(data, dict) or ("run_mode" not in data and "mode" not in data):
        return Validated(False, "config.json missing run_mode/mode", details)

    mode_val = data.get("mode") or data.get("run_mode") or ""
    skipped = (
        bool(data.get("skipped"))
        or mode_val == "simulated_participants_nobrowser"
        or bool(data.get("ground_truth_model"))
    )
    if skipped:
        details["skipped"] = True
        return Validated(
            True, "Implement output valid (skip-deploy mode; config.json only)", details
        )

    if not index_path.exists():
        return Validated(False, "index.html not found", {"path": str(index_path)})
    text = index_path.read_text(encoding="utf-8")
    if "jsPsych" not in text and "jspsych" not in text.lower():
        return Validated(False, "index.html does not mention jsPsych", details)
    return Validated(True, "Implement output valid", details)
