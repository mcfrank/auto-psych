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
        # A missing registry legitimately means "no theories accumulated yet".
        return {"theories": {}, "reserved_for_new": DEFAULT_RESERVED_FOR_NEW}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        # A registry that exists but cannot be parsed is a corruption, NOT an
        # empty registry. Returning a default here would silently discard every
        # accumulated theory probability; fail loudly instead.
        raise ValueError(f"Could not parse model registry at {path}: {exc}") from exc
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


def normalize_theories(
    theories: Dict[str, float], reserved: float = 0.0
) -> Dict[str, float]:
    """Scale theory probabilities so they sum to (1 - reserved).

    When the weights sum to <= 0 (an all-zero or collapsed registry) there is no
    mass to scale, so redistribute the target uniformly rather than returning all
    zeros. (Do NOT coalesce the sum to 1.0 first — that would mask the zero-sum
    case and silently emit an all-zero distribution.)
    """
    total = sum(theories.values())
    target = max(0.0, 1.0 - reserved)
    if total <= 0:
        n = len(theories) or 1
        return {k: target / n for k in theories}
    return {k: (v / total) * target for k, v in theories.items()}
