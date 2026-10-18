"""Final-digit similarity + common-factor rule + decade-span coverage.

A refinement of digit_similarity_with_common_factor: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1, same
parity other digit = partial, opposite parity = 0) plus an inferred boost for
multiples of the examples' common factor (>= 3). The one change: people code
magnitude by decade and treat every decade from the examples' lowest to their
highest as covered, so a target whose tens digit lies inside that decade span
gets the full magnitude boost, which fades with each decade beyond the span.
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
    decades = [x // 10 for x in xs]
    td = t // 10
    beyond = max(min(decades) - td, td - max(decades), 0)
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": divisible,
        "decades_beyond_span": float(beyond),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    beyond = pm.Data("decades_beyond_span", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    # Coverage of the examples' decade span, fading per decade beyond it.
    span_boost = pm.HalfNormal("span_boost", sigma=2.0)
    span_width = pm.LogNormal("span_width", mu=np.log(1.5), sigma=0.75)
    coverage = pm.math.exp(-beyond / span_width)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + factor_boost * cf_mult
            + span_boost * coverage
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
