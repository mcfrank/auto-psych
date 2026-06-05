"""Validation dispatch for pipeline agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.runtime.config import run_dir_for_state
from src.runtime.observability import append_validation_failure
from src.validation.stages.analysis import validate_analyst_output
from src.validation.stages.collect import validate_collect_output
from src.validation.stages.design import validate_designer_output
from src.validation.stages.implement import validate_implementer_output
from src.validation.stages.interpret import validate_interpreter_output
from src.validation.stages.theory import validate_theorist_output
from src.validation.types import Validated


AGENT_VALIDATORS = {
    "1_theory": validate_theorist_output,
    "2_design": validate_designer_output,
    "3_implement": validate_implementer_output,
    "4_collect": validate_collect_output,
    "5_analyze": validate_analyst_output,
    "6_interpret": validate_interpreter_output,
}

NODE_TO_AGENT_KEY = {
    "theory": "1_theory",
    "design": "2_design",
    "implement": "3_implement",
    "collect": "4_collect",
    "analyze": "5_analyze",
    "interpret": "6_interpret",
}


def run_validation(state: Dict[str, Any], agent_key: str) -> Dict[str, Any]:
    """Run the validator for the given agent on the current run directory."""
    project_id = state.get("project_id", "")
    run_id = state.get("run_id", 0)
    rdir = run_dir_for_state(project_id, run_id, state)
    validator_fn = AGENT_VALIDATORS.get(agent_key)
    if not validator_fn:
        return {
            **state,
            "validation_ok": True,
            "validation_feedback": "",
            "validation_retry_count": 0,
            "last_validated_agent": agent_key,
        }

    v = validator_fn(rdir)
    retry_count = state.get("validation_retry_count", 0)
    if v.ok:
        return {
            **state,
            "validation_ok": True,
            "validation_feedback": "",
            "validation_retry_count": 0,
            "last_validated_agent": agent_key,
        }

    try:
        append_validation_failure(
            Path(rdir),
            agent_key,
            attempt=retry_count + 1,
            message=v.message,
            details=v.details,
        )
    except Exception:
        pass

    return {
        **state,
        "validation_ok": False,
        "validation_feedback": v.message
        + (f" Details: {v.details}" if v.details else ""),
        "validation_retry_count": retry_count + 1,
        "last_validated_agent": agent_key,
    }
