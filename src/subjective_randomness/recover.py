"""Ground-truth sampling and recovery summaries for parameter recovery.

Shared by every Bayesian recovery path — the PyMC parameter recovery
(:mod:`src.subjective_randomness.pymc_recover`) and the grid-posterior
stimulus-selection comparison
(:mod:`src.subjective_randomness.adaptive_recovery`). Sampled-truth recovery
draws each repeat's ground-truth vector uniformly from the model family's
``PARAM_BOUNDS`` (optionally narrowed by a ``param_ranges`` config entry), so
recovery is evaluated across the parameter space rather than at one
hand-picked point.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Mapping, Sequence, Tuple

# Offset added to the config seed for ground-truth draws, so truth sampling
# never shares a stream with simulation (seed + repeat) or fitting
# (seed + 10000 + repeat).
TRUTH_SEED_OFFSET = 20000


def resolve_param_ranges(
    config: Mapping[str, Any], model_module
) -> Dict[str, Tuple[float, float]]:
    """Ranges to sample ground-truth parameters from.

    Defaults to the model family's ``PARAM_BOUNDS``; a ``param_ranges`` config
    entry may narrow individual parameters. Unknown parameter names, inverted
    ranges, and ranges reaching outside the fit bounds (truths there could
    never be recovered) all fail loudly.
    """
    ranges = {
        name: (float(lo), float(hi))
        for name, (lo, hi) in getattr(model_module, "PARAM_BOUNDS").items()
    }
    for name, value in (config.get("param_ranges") or {}).items():
        if name not in ranges:
            raise ValueError(
                f"param_ranges names unknown parameter {name!r}; "
                f"{model_module.__name__} has parameters {sorted(ranges)}."
            )
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(
                f"param_ranges for {name!r} must be a [low, high] pair, got {value!r}."
            )
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            raise ValueError(f"param_ranges for {name!r} is inverted: [{lo}, {hi}].")
        bound_lo, bound_hi = ranges[name]
        if lo < bound_lo or hi > bound_hi:
            raise ValueError(
                f"param_ranges for {name!r} ([{lo}, {hi}]) extends outside the fit "
                f"bounds [{bound_lo}, {bound_hi}]; truths there can never be recovered."
            )
        ranges[name] = (lo, hi)
    return ranges


def sample_true_params(
    param_ranges: Mapping[str, Tuple[float, float]], rng: random.Random
) -> Dict[str, float]:
    """Draw one ground-truth parameter vector uniformly from `param_ranges`."""
    return {
        name: round(rng.uniform(lo, hi), 6) for name, (lo, hi) in param_ranges.items()
    }


def _pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` when undefined (n < 2 or zero variance).

    Variance is judged by distinct values, not the moment sums: a constant
    value like 0.4 accumulates ~1e-17 of float noise in the arithmetic and
    would otherwise yield a garbage near-zero correlation instead of
    "undefined".
    """
    n = len(xs)
    if n < 2 or len(set(xs)) == 1 or len(set(ys)) == 1:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def summarize_paired_recovery(
    truths: Sequence[Mapping[str, float]], estimates: Sequence[Mapping[str, float]]
) -> Dict[str, Any]:
    """Per-parameter recovery quality from paired (truth, estimate) runs."""
    summary: Dict[str, Any] = {}
    names = sorted({name for truth in truths for name in truth})
    for name in names:
        pairs = [
            (float(truth[name]), float(est[name]))
            for truth, est in zip(truths, estimates)
            if name in truth and name in est
        ]
        if not pairs:
            continue
        trues = [t for t, _ in pairs]
        ests = [e for _, e in pairs]
        errors = [e - t for t, e in pairs]
        constant_truth = len(set(trues)) == 1
        pearson = _pearson_r(trues, ests)
        summary[name] = {
            "true": trues[0] if constant_truth else None,
            "mean_estimate": round(sum(ests) / len(ests), 6),
            "bias": round(sum(errors) / len(errors), 6),
            "rmse": round(math.sqrt(sum(e**2 for e in errors) / len(errors)), 6),
            "min_estimate": round(min(ests), 6),
            "max_estimate": round(max(ests), 6),
            "pearson_r": None if pearson is None else round(pearson, 6),
        }
    return summary
