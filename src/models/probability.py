"""Validation helpers for probabilities crossing model boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np


def validate_probability(value: Any, *, context: str) -> float:
    """Return ``value`` as a float after enforcing a finite [0, 1] contract."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} must be a numeric probability, got {value!r}.")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{context} must be finite and in [0, 1], got {probability!r}."
        )
    return probability


def validate_probability_distribution(
    distribution: Any,
    response_options: Sequence[str],
    *,
    context: str,
    tolerance: float = 1e-5,
) -> dict[str, float]:
    """Validate an exact categorical distribution over ``response_options``."""
    if not isinstance(distribution, Mapping):
        raise TypeError(f"{context} must return a probability mapping.")
    options = list(response_options)
    if not options or len(set(options)) != len(options):
        raise ValueError(f"{context}: response options must be non-empty and unique.")
    missing = set(options) - set(distribution)
    extra = set(distribution) - set(options)
    if missing or extra:
        raise ValueError(
            f"{context} returned the wrong response keys; missing={sorted(missing)}, "
            f"extra={sorted(extra)}. Expected exactly {options}."
        )
    validated = {
        option: validate_probability(
            distribution[option], context=f"{context} probability for {option!r}"
        )
        for option in options
    }
    total = math.fsum(validated.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{context} probabilities must sum to 1, got {total}.")
    return validated


def validate_probability_array(values: Any, *, context: str) -> np.ndarray:
    """Return a float array after enforcing finite values in [0, 1]."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be a numeric probability array.") from exc
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{context} values must be finite and in [0, 1].")
    return array
