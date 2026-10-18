"""Example-count-dependent neighbourhood reliance on the incumbent digit model.

A refinement of final_digit_rule_suppresses_parity: "yes" still grows with the
target's average final-digit similarity to the examples (parity credit
suppressed when the examples share one final digit), plus a boost for
multiples of the examples' common factor (>= 3) and decade-band familiarity.
The one change: the weight on decade-band familiarity is not fixed but falls
with the number of examples shown (weight = band_boost * n ** -band_decay).
A lone example offers no shared pattern, so people generalize mostly to its
decade; with more examples they rely on the shared digit/factor patterns.
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
        "log_n_examples": math.log(n),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    conc = pm.Data("final_digit_concentration", np.zeros(1, dtype="float64"))
    log_n = pm.Data("log_n_examples", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    rule_suppression = pm.Beta("rule_suppression", alpha=1.5, beta=1.5)
    parity_credit = s_parity * (1.0 - rule_suppression * conc)
    mean_sim = frac_digit + parity_credit * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    # How fast reliance on the decade neighbourhood fades as examples accumulate.
    band_decay = pm.Normal("band_decay", mu=0.0, sigma=1.0)
    band_sim = pm.math.exp(-band_gap / band_width)
    band_weight = band_boost * pm.math.exp(-band_decay * log_n)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + factor_boost * cf_mult
            + band_weight * band_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
