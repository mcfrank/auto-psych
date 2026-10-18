"""Rule-less example sets invite an "anything goes" belief.

A refinement of rule_gated_decade_familiarity: "yes" still grows with the
target's average final-digit similarity to the examples (an abstracted
"ends in d" rule removing an inferred share of parity credit), a boost for
multiples of the examples' common factor (>= 3) and decade-band familiarity
cut by an inferred share when an explicit rule is detected. The one change:
when two or more examples show no detectable rule (mixed final digits, no
common factor >= 3), people believe with an inferred probability that the
program produces any number, and under that belief say "yes" whatever the
target -- a set-type-gated acceptance floor, p = w + (1 - w) * p_rule.
Lone examples and rule-bearing sets keep the incumbent's judgments.
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
    uniform = len(xs) >= 2 and len({x % 10 for x in xs}) == 1
    rule_found = len(xs) >= 2 and (uniform or g >= 3)
    ruleless_multi = len(xs) >= 2 and not rule_found
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": divisible,
        "decade_band_gap": float(band_gap),
        "shared_digit_rule": 1.0 if uniform else 0.0,
        "explicit_rule_detected": 1.0 if rule_found else 0.0,
        "ruleless_multi_set": 1.0 if ruleless_multi else 0.0,
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    rule = pm.Data("shared_digit_rule", np.zeros(1, dtype="float64"))
    any_rule = pm.Data("explicit_rule_detected", np.zeros(1, dtype="float64"))
    ruleless = pm.Data("ruleless_multi_set", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    preempt = pm.Beta("preempt", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * (1.0 - preempt * rule) * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)
    magnitude_preempt = pm.Beta("magnitude_preempt", alpha=2.0, beta=2.0)

    p_rule = pm.math.sigmoid(
        intercept + gain * mean_sim + factor_boost * cf_mult
        + band_boost * (1.0 - magnitude_preempt * any_rule) * band_sim
    )
    # Probability of the "program produces anything" belief for rule-less sets.
    anything_goes = pm.Beta("anything_goes", alpha=1.5, beta=4.0)
    w = anything_goes * ruleless
    p_yes = pm.Deterministic(
        "p_yes", pm.math.clip(w + (1.0 - w) * p_rule, 1e-6, 1.0 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
