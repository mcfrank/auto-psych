"""
Validators that check agent outputs meet minimal standards.
Return Validated(ok, message, details) for observability and tests.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.models.loader import get_model_callable, get_model_names_from_manifest
from src.models.randomness import MODEL_LIBRARY


@dataclass
class Validated:
    ok: bool
    message: str
    details: Optional[Dict[str, Any]] = None


def validate_theorist_output(run_dir: Path) -> Validated:
    """
    (1) models_manifest.yaml exists and is valid YAML;
    (2) every model name is loadable: either <name>.py exists in 1_theory/ or name is in MODEL_LIBRARY;
    (3) for each model, call with a test stimulus and check it returns a dict
        mapping response options to probabilities that sum to 1.
    """
    run_dir = Path(run_dir)
    theorist_dir = run_dir / "1_theory"
    manifest_path = theorist_dir / "models_manifest.yaml"
    details: Dict[str, Any] = {}

    if not manifest_path.exists():
        return Validated(False, "models_manifest.yaml not found", {"path": str(manifest_path)})

    try:
        raw = manifest_path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except Exception as e:
        return Validated(False, f"Invalid YAML: {e}", {"path": str(manifest_path)})

    if not isinstance(data, dict):
        return Validated(False, "manifest is not a dict", details)

    models = data.get("models") or []
    if not isinstance(models, list):
        return Validated(False, "manifest 'models' is not a list", details)

    names = []
    for m in models:
        if isinstance(m, dict) and "name" in m:
            names.append(m["name"])
        elif isinstance(m, str):
            names.append(m)
    details["model_names"] = names

    # Resolve loadable names: run-dir .py or MODEL_LIBRARY
    loadable = get_model_names_from_manifest(data, theorist_dir)
    for name in names:
        if name not in loadable:
            return Validated(
                False,
                f"Model '{name}' has no 1_theory/{name}.py and is not in MODEL_LIBRARY",
                details,
            )

    # Test stimulus: each model must return a probability distribution
    test_stimulus = ("HHTHTTHT", "HTHTHTHT")
    response_options = ["left", "right"]
    for name in names:
        try:
            fn = get_model_callable(name, theorist_dir)
            preds = fn(test_stimulus, response_options)
        except Exception as e:
            return Validated(False, f"Model '{name}' raised: {e}", details)
        if not isinstance(preds, dict):
            return Validated(False, f"Model '{name}' did not return a dict", details)
        for k in response_options:
            if k not in preds:
                return Validated(False, f"Model '{name}' missing key '{k}'", details)
        total = sum(preds[k] for k in response_options)
        if abs(total - 1.0) > 1e-5:
            return Validated(False, f"Model '{name}' probabilities sum to {total}", details)

    return Validated(True, "Theorist output valid; all models run and return scores", details)


def validate_designer_output(run_dir: Path) -> Validated:
    """stimuli.json exists, is a list, each item has sequence_a and sequence_b."""
    run_dir = Path(run_dir)
    path = run_dir / "2_design" / "stimuli.json"
    details: Dict[str, Any] = {}

    if not path.exists():
        return Validated(False, "stimuli.json not found", {"path": str(path)})

    try:
        raw = path.read_text(encoding="utf-8")
        data = __import__("json").loads(raw)
    except Exception as e:
        return Validated(False, f"Invalid JSON: {e}", {"path": str(path)})

    if not isinstance(data, list):
        return Validated(False, "stimuli.json is not a list", details)

    eig_values = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return Validated(False, f"Stimulus {i} is not a dict", {"index": i})
        if "sequence_a" not in item or "sequence_b" not in item:
            return Validated(False, f"Stimulus {i} missing sequence_a or sequence_b", {"index": i})
        if "eig" not in item:
            return Validated(False, f"Stimulus {i} missing required 'eig' field (EIG value for observability)", {"index": i})
        try:
            eig_values.append(float(item["eig"]))
        except (TypeError, ValueError):
            return Validated(False, f"Stimulus {i} has non-numeric 'eig'", {"index": i})

    details["n_stimuli"] = len(data)
    if not eig_values:
        return Validated(True, f"Designer output valid; {len(data)} stimuli (no eig)", details)
    details["eig_min"] = min(eig_values)
    details["eig_max"] = max(eig_values)
    if max(eig_values) <= 0:
        return Validated(False, "All EIG values are <= 0; check design script (stimulus must be tuple for get_model_predictions)", details)
    # Treat fallback design as validation failure so the pipeline retries the designer with feedback
    rationale_path = run_dir / "2_design" / "design_rationale.md"
    if rationale_path.exists():
        rationale_preview = rationale_path.read_text(encoding="utf-8").strip()[:200]
        if rationale_preview.startswith("Fallback design:"):
            return Validated(
                False,
                "Design was generated by the fallback (no LLM script was used or script failed). "
                "You must output exactly one fenced ```python block containing the full design script. "
                "Do not implement EIG yourself; call expected_information_gain((seq_a, seq_b)). "
                "Each stimulus in stimuli.json must include an 'eig' field.",
                {**details, "fallback_used": True},
            )
    return Validated(True, f"Designer output valid; {len(data)} stimuli, EIG range [{min(eig_values):.4f}, {max(eig_values):.4f}]", details)


def validate_implementer_output(run_dir: Path) -> Validated:
    """index.html exists with jsPsych; config.json exists (deploy step bundled)."""
    run_dir = Path(run_dir)
    impl_dir = run_dir / "3_implement"
    index_path = impl_dir / "index.html"
    config_path = impl_dir / "config.json"
    details: Dict[str, Any] = {}

    if not index_path.exists():
        return Validated(False, "index.html not found", {"path": str(index_path)})
    text = index_path.read_text(encoding="utf-8")
    if "jsPsych" not in text and "jspsych" not in text.lower():
        return Validated(False, "index.html does not mention jsPsych", details)
    if not config_path.exists():
        return Validated(False, "config.json not found (deploy step)", {"path": str(config_path)})
    try:
        data = __import__("json").loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        return Validated(False, f"Invalid config.json: {e}", {"path": str(config_path)})
    if not isinstance(data, dict) or ("run_mode" not in data and "mode" not in data):
        return Validated(False, "config.json missing run_mode/mode", details)
    return Validated(True, "Implement output valid", details)


def validate_collect_output(run_dir: Path) -> Validated:
    """responses.csv exists and has expected columns."""
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


def validate_analyst_output(run_dir: Path) -> Validated:
    """aggregate.csv exists with expected columns; summary_stats.json is valid JSON with expected keys."""
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
        return Validated(False, "aggregate.csv missing expected columns", {"header": list(header)})

    if not summary_path.exists():
        return Validated(False, "summary_stats.json not found", {"path": str(summary_path)})
    try:
        data = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        return Validated(False, f"Invalid summary_stats.json: {e}", details)
    if not isinstance(data, dict):
        return Validated(False, "summary_stats.json is not a dict", details)

    return Validated(True, "Data analyst output valid", details)


def validate_interpreter_output(run_dir: Path) -> Validated:
    """report.md exists and is non-empty."""
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


# Map agent_key -> validator function for critic CLI
AGENT_VALIDATORS = {
    "1_theory": validate_theorist_output,
    "2_design": validate_designer_output,
    "3_implement": validate_implementer_output,
    "4_collect": validate_collect_output,
    "5_analyze": validate_analyst_output,
    "6_interpret": validate_interpreter_output,
}

# Map graph node name -> agent_key for validation loop
NODE_TO_AGENT_KEY = {
    "theory": "1_theory",
    "design": "2_design",
    "implement": "3_implement",
    "collect": "4_collect",
    "analyze": "5_analyze",
    "interpret": "6_interpret",
}


def run_validation(state: Dict[str, Any], agent_key: str) -> Dict[str, Any]:
    """
    Run the validator for the given agent on the current run directory.
    Return state with validation_ok, validation_feedback, and validation_retry_count updated.
    """
    from src.config import run_dir
    project_id = state.get("project_id", "")
    run_id = state.get("run_id", 0)
    rdir = run_dir(project_id, run_id)
    validator_fn = AGENT_VALIDATORS.get(agent_key)
    if not validator_fn:
        return {
            **state,
            "validation_ok": True,
            "validation_feedback": "",
            "validation_retry_count": 0,
        }
    v = validator_fn(rdir)
    retry_count = state.get("validation_retry_count", 0)
    if v.ok:
        return {
            **state,
            "validation_ok": True,
            "validation_feedback": "",
            "validation_retry_count": 0,
        }
    return {
        **state,
        "validation_ok": False,
        "validation_feedback": v.message + (f" Details: {v.details}" if v.details else ""),
        "validation_retry_count": retry_count + 1,
    }
