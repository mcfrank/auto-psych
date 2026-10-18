"""Decade-band digit similarity with familiarity fading on a log number line.

A refinement of decade_band_digit_common_factor: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1, same
parity other digit = partial, opposite parity = 0), plus an inferred boost for
multiples of the examples' common factor (>= 3) and full familiarity anywhere
in the band of decades the examples span. The one change: familiarity fades
outside the band with distance on a compressed, logarithmic mental number line
(log ratio of the target to the band's nearer edge), not by decades counted,
so a target above the band is more familiar than one equally far below it.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = float(len(xs))
    same_digit = sum(1 for x in xs if x % 10 == t % 10) / n
    same_parity_other = sum(1 for x in xs if x % 2 == t % 2 and x % 10 != t % 10) / n
    g = reduce(math.gcd, xs)
    divisible = 1.0 if (g >= 3 and t % g == 0) else 0.0
    # Band edges: start of the lowest example's decade, end of the highest's.
    lo_edge = max(10 * min(x // 10 for x in xs), 1)
    hi_edge = 10 * max(x // 10 for x in xs) + 9
    log_gap = max(math.log(lo_edge / t), 0.0, math.log(t / hi_edge))
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": divisible,
        "log_band_gap": float(log_gap),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    log_gap = pm.Data("log_band_gap", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    # Familiarity: full inside the band, decaying with log-ratio distance.
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(0.5), sigma=0.75)
    band_sim = pm.math.exp(-log_gap / band_width)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + factor_boost * cf_mult
            + band_boost * band_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
