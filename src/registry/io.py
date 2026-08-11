"""
Per-run model registry: theories and their probabilities (sum to 1).
Schema: { "theories": { "model_name": float }, "reserved_for_new": float }
"""

import math
from numbers import Real
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
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        # A registry that exists but cannot be parsed is a corruption, NOT an
        # empty registry. Returning a default here would silently discard every
        # accumulated theory probability; fail loudly instead.
        raise ValueError(f"Could not parse model registry at {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Model registry at {path} must be a mapping, got "
            f"{type(data).__name__}."
        )
    if "theories" in data:
        theories = data["theories"]
    elif "probabilities" in data:
        theories = data["probabilities"]
    else:
        theories = {}
    # Coercing a malformed block to a default is the same silent-data-loss bug
    # as swallowing the parse error above: it would hand the designer a prior
    # that no experiment produced.
    if not isinstance(theories, dict):
        raise ValueError(
            f"Model registry at {path} has a malformed `theories` block: expected "
            f"a mapping of model_name -> probability, got {type(theories).__name__}."
        )
    validated_theories = validate_theory_weights(theories, context=f"registry at {path}")
    validated_reserved = _validate_reserved(
        data.get("reserved_for_new", DEFAULT_RESERVED_FOR_NEW),
        context=f"registry at {path}",
    )
    return {"theories": validated_theories, "reserved_for_new": validated_reserved}


def validate_theory_weights(
    theories: Dict[str, float], *, context: str = "model weights"
) -> Dict[str, float]:
    """Validate finite, non-negative weights keyed by non-empty model names."""
    if not isinstance(theories, dict):
        raise ValueError(f"{context} theories must be a mapping.")
    validated: Dict[str, float] = {}
    for name, value in theories.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{context} contains an invalid model name {name!r}.")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{context} weight for {name!r} must be numeric, got {value!r}.")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"{context} weight for {name!r} must be finite and non-negative, "
                f"got {weight!r}."
            )
        validated[name] = weight
    return validated


def _validate_reserved(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"{context} has malformed `reserved_for_new`: expected a number, "
            f"got {value!r}."
        )
    reserved = float(value)
    if not math.isfinite(reserved) or not 0.0 <= reserved <= 1.0:
        raise ValueError(
            f"{context} `reserved_for_new` must be finite and in [0, 1], got "
            f"{reserved!r}."
        )
    return reserved


def write_registry(
    registry_path: Path,
    theories: Dict[str, float],
    reserved_for_new: float = DEFAULT_RESERVED_FOR_NEW,
) -> None:
    """Write model_registry.yaml. theories map model_name -> probability."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "theories": validate_theory_weights(theories),
        "reserved_for_new": _validate_reserved(
            reserved_for_new, context="model registry"
        ),
    }
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
    theories = validate_theory_weights(theories)
    reserved = _validate_reserved(reserved, context="model weights")
    total = math.fsum(theories.values())
    target = 1.0 - reserved
    if total <= 0:
        n = len(theories) or 1
        return {k: target / n for k in theories}
    return {k: (v / total) * target for k, v in theories.items()}
