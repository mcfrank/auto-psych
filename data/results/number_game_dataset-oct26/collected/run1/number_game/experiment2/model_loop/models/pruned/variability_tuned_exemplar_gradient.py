"""Variability-tuned exemplar generalization over final digits.

People judge the target by its similarity to the stored examples, and the
breadth of generalization around those exemplars is set by how variable the
exemplars are (category-variability effect): when the examples all share one
final digit the final-digit similarity gradient is tight, so a target sharing
only their parity gets little credit; when the examples' final digits vary the
gradient is broad and same-parity targets get substantial credit. The rest of
the comparison follows decade_band_digit_common_factor: average final-digit
similarity to the examples, decade-band familiarity, common-factor boost.
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
    # Perceived variability of the exemplars' final digits: share of examples
    # beyond the first that add a new digit, shrunk with half a pseudo-example
    # so a lone example reads as intermediate (0.5).
    distinct = len(set(x % 10 for x in xs))
    digit_spread = (distinct - 1 + 0.5) / n
    g = reduce(math.gcd, xs)
    divisible = 1.0 if (g >= 3 and t % g == 0) else 0.0
    lo = min(x // 10 for x in xs)
    hi = max(x // 10 for x in xs)
    td = t // 10
    band_gap = max(lo - td, 0, td - hi)
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "digit_spread": float(digit_spread),
        "common_factor_multiple": divisible,
        "decade_band_gap": float(band_gap),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    spread = pm.Data("digit_spread", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))

    # Same-parity similarity widens with exemplar variability (centred at
    # spread = 0.5, the lone-example value).
    par_logit = pm.Normal("par_logit", mu=0.0, sigma=1.5)
    spread_slope = pm.HalfNormal("spread_slope", sigma=5.0)
    s_parity = pm.math.sigmoid(par_logit + spread_slope * (spread - 0.5))
    mean_sim = frac_digit + s_parity * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + factor_boost * cf_mult
            + band_boost * band_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
