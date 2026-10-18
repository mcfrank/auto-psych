"""Final-digit similarity plus a common-factor (multiples) rule.

A refinement of final_digit_similarity_gradient: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1, same
parity other digit = partial, opposite parity = 0), but the one change is that
people also notice the greatest common divisor (>= 3) of the examples and give
an inferred extra boost to targets that are multiples of it, regardless of
their final digit.
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
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": divisible,
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # Extra log-odds of yes for a multiple of the examples' common factor.
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + gain * mean_sim + factor_boost * cf_mult)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
