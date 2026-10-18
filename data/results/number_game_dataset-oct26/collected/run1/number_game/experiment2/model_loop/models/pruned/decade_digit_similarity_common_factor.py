"""Final-digit plus tens-digit (decade) similarity, with a common-factor rule.

A refinement of digit_similarity_with_common_factor: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1, same
parity other digit = partial, opposite parity = 0) and with an inferred boost
for multiples of the examples' common divisor (>= 3). The one change is that
people also compare the target's tens digit (decade) with each example's: same
decade = 1, adjacent decade = partial (inferred), otherwise 0, and "yes" also
grows with this average decade similarity.
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
    same_decade = sum(1 for x in xs if x // 10 == t // 10) / n
    adj_decade = sum(1 for x in xs if abs(x // 10 - t // 10) == 1) / n
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": divisible,
        "frac_same_decade": same_decade,
        "frac_adjacent_decade": adj_decade,
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    frac_dec = pm.Data("frac_same_decade", np.zeros(1, dtype="float64"))
    frac_adj = pm.Data("frac_adjacent_decade", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    # Decade similarity: same tens digit = 1, adjacent tens digit = s_adjacent.
    s_adjacent = pm.Beta("s_adjacent", alpha=2.0, beta=2.0)
    decade_sim = frac_dec + s_adjacent * frac_adj

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    decade_gain = pm.HalfNormal("decade_gain", sigma=3.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + decade_gain * decade_sim + factor_boost * cf_mult
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
