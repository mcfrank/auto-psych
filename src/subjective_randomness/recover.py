"""Pearson correlation for recovery metrics."""

from __future__ import annotations

import math
from typing import Sequence


def pearson_r(xs: Sequence[float], ys: Sequence[float]) -> float | None:
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
