"""Final-digit similarity + common-factor rule + span-filled decade familiarity.

A refinement of nearest_decade_digit_common_factor: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1, same
parity other digit = partial, opposite parity = 0), plus an inferred boost for
multiples of the examples' common factor (>= 3) and a leading-digit (decade)
familiarity boost that decays with decade separation. The one change: people
fill in the decades between the smallest and largest example, so the decade
boost is full anywhere within the examples' decade span and decays only with
how many decades the target lies beyond that span.
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
    lo, hi = min(xs) // 10, max(xs) // 10
    td = t // 10
    span_decade_gap = max(lo - td, td - hi, 0)
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": divisible,
        "span_decade_gap": float(span_decade_gap),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    dec_gap = pm.Data("span_decade_gap", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    # Decade familiarity: full inside the examples' decade span, decaying beyond it.
    decade_boost = pm.HalfNormal("decade_boost", sigma=2.0)
    decade_width = pm.LogNormal("decade_width", mu=np.log(1.5), sigma=0.75)
    decade_sim = pm.math.exp(-dec_gap / decade_width)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + factor_boost * cf_mult
            + decade_boost * decade_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
