"""Attentional lapses with a biased guess, on top of rule-gated digit similarity.

People judge the target as rule_gated_decade_familiarity does (final-digit
similarity averaged over the examples, parity credit pre-empted by a shared
final-digit rule, a common-factor boost, decade-band familiarity pre-empted by
a detected rule). The one change is in the decision rule: on an inferred share
of trials attention lapses and they answer by an unconsidered guess, "yes"
with an inferred guessing bias, regardless of set and target. This bounds
p(yes) away from 0 and 1 by a floor and ceiling the logistic cannot produce.
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
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": divisible,
        "decade_band_gap": float(band_gap),
        "shared_digit_rule": 1.0 if uniform else 0.0,
        "explicit_rule_detected": 1.0 if rule_found else 0.0,
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    rule = pm.Data("shared_digit_rule", np.zeros(1, dtype="float64"))
    any_rule = pm.Data("explicit_rule_detected", np.zeros(1, dtype="float64"))

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

    p_judged = pm.math.sigmoid(
        intercept + gain * mean_sim + factor_boost * cf_mult
        + band_boost * (1.0 - magnitude_preempt * any_rule) * band_sim
    )

    # Decision rule: share of lapsed trials, and the guess's bias toward "yes".
    lapse = pm.Beta("lapse", alpha=1.5, beta=10.0)
    guess_yes = pm.Beta("guess_yes", alpha=2.0, beta=2.0)

    p_yes = pm.Deterministic(
        "p_yes", (1.0 - lapse) * p_judged + lapse * guess_yes
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
