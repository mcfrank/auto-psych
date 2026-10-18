"""Coincidence-graded rule credence pre-empts decade-band familiarity.

A refinement of rule_gated_decade_familiarity: "yes" still grows with the
target's average final-digit similarity to the examples (an abstracted
"ends in d" rule removes an inferred share of parity credit), plus a boost for
multiples of the examples' common factor (>= 3) and decade-band familiarity
decaying outside the band. The one change: instead of a binary "explicit rule
detected" switch, belief in a rule is graded by how improbable the examples'
shared property is by coincidence -- log-evidence (n - 1) * log(1/p) with
p = 1/10 for a shared final digit, 1/g for a common factor g, 1/2 for a shared
parity (the strongest such property counts). Rule credence
1 - exp(-k * evidence) is the share of magnitude familiarity pre-empted.
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
    # Coincidence log-evidence for the strongest property all examples share.
    per_example = 0.0
    if len(xs) >= 2:
        if uniform:
            per_example = max(per_example, math.log(10.0))
        if g >= 2:
            per_example = max(per_example, math.log(float(g)))
        if len({x % 2 for x in xs}) == 1:
            per_example = max(per_example, math.log(2.0))
    evidence = (n - 1.0) * per_example
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": divisible,
        "decade_band_gap": float(band_gap),
        "shared_digit_rule": 1.0 if uniform else 0.0,
        "rule_coincidence_evidence": evidence,
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    rule = pm.Data("shared_digit_rule", np.zeros(1, dtype="float64"))
    evidence = pm.Data("rule_coincidence_evidence", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    preempt = pm.Beta("preempt", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * (1.0 - preempt * rule) * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)
    # How fast coincidence evidence turns into belief in a rule.
    credence_rate = pm.LogNormal("credence_rate", mu=np.log(0.3), sigma=1.0)
    rule_credence = 1.0 - pm.math.exp(-credence_rate * evidence)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + factor_boost * cf_mult
            + band_boost * (1.0 - rule_credence) * band_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
