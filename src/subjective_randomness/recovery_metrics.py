"""Pure metric functions for holdout-recovery evaluation.

All functions take ground-truth probabilities ``q`` and recovered
probabilities ``p`` as equal-length sequences, and return a scalar.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple


def _validate_lengths(q: Sequence[float], p: Sequence[float]) -> int:
    if len(q) != len(p):
        raise ValueError(
            f"q and p must have the same length, got {len(q)} and {len(p)}"
        )
    return len(q)


def kl_regret(
    q: Sequence[float], p: Sequence[float], eps: float = 1e-9
) -> float:
    """Expected Bernoulli KL divergence from the ground truth to the estimate.

    ``KL(q || p) = q·log(q/p) + (1−q)·log((1−q)/(1−p))``, averaged over
    stimuli. Both ``q`` and ``p`` are clipped to ``[eps, 1−eps]``.
    """
    n = _validate_lengths(q, p)
    total = 0.0
    for qi, pi in zip(q, p):
        qi_c = max(eps, min(1.0 - eps, qi))
        pi_c = max(eps, min(1.0 - eps, pi))
        total += qi_c * math.log(qi_c / pi_c) + (1.0 - qi_c) * math.log(
            (1.0 - qi_c) / (1.0 - pi_c)
        )
    return total / n


def bias(q: Sequence[float], p: Sequence[float]) -> float:
    """Mean signed error: ``mean(p − q)``."""
    n = _validate_lengths(q, p)
    return sum(pi - qi for qi, pi in zip(q, p)) / n


def calibration(
    q: Sequence[float], p: Sequence[float]
) -> Tuple[float, float]:
    """OLS regression of ``p`` on ``q``: returns ``(slope, intercept)``."""
    n = _validate_lengths(q, p)
    mean_q = sum(q) / n
    mean_p = sum(p) / n
    ss_qq = sum((qi - mean_q) ** 2 for qi in q)
    if ss_qq == 0.0:
        return (0.0, mean_p)
    cov_qp = sum((qi - mean_q) * (pi - mean_p) for qi, pi in zip(q, p))
    slope = cov_qp / ss_qq
    intercept = mean_p - slope * mean_q
    return (slope, intercept)


def rmse(q: Sequence[float], p: Sequence[float]) -> float:
    """Root mean squared error: ``sqrt(mean((p − q)²))``."""
    n = _validate_lengths(q, p)
    return math.sqrt(sum((pi - qi) ** 2 for qi, pi in zip(q, p)) / n)
