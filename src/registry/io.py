"""
Per-run model registry: theories and their probabilities (sum to 1).
Schema: { "theories": { "model_name": float }, "reserved_for_new": float }
"""

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_RESERVED_FOR_NEW = 0.25


def load_registry(registry_path: Path) -> Dict[str, Any]:
    """Load model_registry.yaml; return dict with 'theories' and 'reserved_for_new'."""
    path = Path(registry_path)
    if not path.exists():
        return {"theories": {}, "reserved_for_new": DEFAULT_RESERVED_FOR_NEW}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"theories": {}, "reserved_for_new": DEFAULT_RESERVED_FOR_NEW}
    theories = data.get("theories") or data.get("probabilities") or {}
    if not isinstance(theories, dict):
        theories = {}
    reserved = data.get("reserved_for_new", DEFAULT_RESERVED_FOR_NEW)
    if not isinstance(reserved, (int, float)):
        reserved = DEFAULT_RESERVED_FOR_NEW
    return {"theories": dict(theories), "reserved_for_new": float(reserved)}


def write_registry(
    registry_path: Path,
    theories: Dict[str, float],
    reserved_for_new: float = DEFAULT_RESERVED_FOR_NEW,
) -> None:
    """Write model_registry.yaml. theories map model_name -> probability."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"theories": theories, "reserved_for_new": reserved_for_new}
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def get_model_weights(registry_path: Path) -> Dict[str, float]:
    """
    Return dict of model_name -> probability for EIG/designer.
    Excludes reserved_for_new; only returns theories. Caller may normalize.
    """
    reg = load_registry(registry_path)
    return dict(reg.get("theories") or {})


def normalize_theories(theories: Dict[str, float], reserved: float = 0.0) -> Dict[str, float]:
    """Scale theory probabilities so they sum to (1 - reserved)."""
    total = sum(theories.values()) or 1.0
    target = max(0.0, 1.0 - reserved)
    if total <= 0:
        n = len(theories) or 1
        return {k: target / n for k in theories}
    return {k: (v / total) * target for k, v in theories.items()}
