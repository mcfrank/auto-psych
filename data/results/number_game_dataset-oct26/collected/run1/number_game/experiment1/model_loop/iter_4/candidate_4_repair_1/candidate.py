"""Final-digit similarity plus common-divisor multiples.

A refinement of final_digit_similarity_gradient: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1, same
parity other digit = partial, opposite parity = 0). The one change: people
notice when all the examples share a common factor of 3 or more (their
greatest common divisor) and give targets that are multiples of that factor an
extra, inferred boost toward "yes".
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
    common_multiple = 1.0 if (g >= 3 and t % g == 0) else 0.0
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_multiple": common_multiple,
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    common_mult = pm.Data("common_multiple", np.zeros(1, dtype="float64"))

    # Similarity of a same-parity, different-digit ending on the digit metric.
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # Extra log-odds for a target that is a multiple of the examples' common factor.
    factor_boost = pm.HalfNormal("factor_boost", sigma=2.0)
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + gain * mean_sim + factor_boost * common_mult)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
