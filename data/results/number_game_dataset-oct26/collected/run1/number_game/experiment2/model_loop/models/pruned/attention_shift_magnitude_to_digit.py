"""Set-size-driven attention shift from magnitude to final digit.

People divide attention between a number's magnitude (its decade) and its final
digit, and the split is set by how many examples they have read: one example
offers no digit regularity to notice, so people judge mostly by magnitude and
readily accept other numbers in that example's decade; each further example
lets digit regularities be detected and draws attention away from magnitude
toward the final digit. Otherwise judgments follow final_digit_rule_suppresses_parity
(final-digit similarity with parity credit suppressed by a shared final digit,
decade-band familiarity, common-factor boost).
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
    lo = min(x // 10 for x in xs)
    hi = max(x // 10 for x in xs)
    td = t // 10
    band_gap = max(lo - td, 0, td - hi)
    digits = [x % 10 for x in xs]
    if len(xs) > 1:
        conc = (max(digits.count(d) for d in digits) - 1) / (n - 1)
    else:
        conc = 0.0
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": divisible,
        "decade_band_gap": float(band_gap),
        "final_digit_concentration": float(conc),
        "extra_examples": n - 1.0,
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    conc = pm.Data("final_digit_concentration", np.zeros(1, dtype="float64"))
    extra = pm.Data("extra_examples", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    rule_suppression = pm.Beta("rule_suppression", alpha=1.5, beta=1.5)
    parity_credit = s_parity * (1.0 - rule_suppression * conc)
    mean_sim = frac_digit + parity_credit * frac_par

    # Attention to magnitude: starts at a1 after one example and moves toward
    # the final digit at an inferred rate with each further example.
    a1 = pm.Beta("attention_magnitude_single", alpha=2.0, beta=2.0)
    shift_rate = pm.HalfNormal("attention_shift_rate", sigma=1.0)
    a_mag = a1 * pm.math.exp(-shift_rate * extra)

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=6.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * ((1.0 - a_mag) * mean_sim + a_mag * band_sim)
            + factor_boost * cf_mult
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
