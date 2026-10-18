"""Decade-band digit similarity + common-factor rule, with diminishing returns.

A refinement of decade_band_digit_common_factor: "yes" still grows with the
target's final-digit similarity to the examples (same digit = 1, same parity
other digit = partial, opposite parity = 0), an inferred boost for multiples
of the examples' common factor (>= 3), and a leading-digit boost that is full
inside the examples' decade band and decays per decade outside it. The one
change: support from resembling examples has diminishing returns. The average
similarity enters through a concave power transform with an inferred exponent,
so the first resembling example counts most and each further one adds less.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = float(len(xs))
    same_digit = sum(1 for x in xs if x % 10 == t % 10) / n
    same_parity_other = sum(1 for x in xs if x % 2 == t % 2 and x % 10 != t % 10) / n
    g = reduce(math.gcd, xs)
    divisible = 1.0 if (g >= 3 and t % g == 0) else 0.0
    lo = min(x // 10 for x in xs)
    hi = max(x // 10 for x in xs)
    td = t // 10
    band_gap = max(lo - td, 0, td - hi)
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": divisible,
        "decade_band_gap": float(band_gap),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par
    # Diminishing returns: concave power transform of the average similarity
    # (exponent 1 = the linear-share model). Small floor keeps gradients finite.
    sim_exponent = pm.LogNormal("sim_exponent", mu=0.0, sigma=0.5)
    support = pt.power(mean_sim + 1e-6, sim_exponent)

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * support + factor_boost * cf_mult
            + band_boost * band_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
