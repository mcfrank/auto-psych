"""Decimal-base factors are more salient rules than other common factors.

A refinement of shortfall_from_mutual_similarity (final-digit similarity, with
only the target's shortfall below the examples' mutual similarity counting
against "yes", rule-pre-empted parity credit and rule-gated decade-band
familiarity, all kept). The one change is to the common-factor boost: when the
examples' common factor is a multiple of 5 (counting by 5s, 10s, 25s -- the
round, base-ten structure of the number system), people find the rule
"multiples of that" far more compelling than a common factor such as 3, 4 or
8, so round targets that fit it (e.g. 100, 80, 50 after 5, 65) get a larger,
separately inferred boost toward "yes" than multiples of a non-decimal factor
(e.g. 64 after 8, 80, 48).
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
    lo = min(x // 10 for x in xs)
    hi = max(x // 10 for x in xs)
    td = t // 10
    uniform = len(xs) >= 2 and len({x % 10 for x in xs}) == 1
    # Mutual final-digit similarity among the examples (all pairs); a lone
    # example has no pair, so no standard is raised.
    pairs = [(a, b) for i, a in enumerate(xs) for b in xs[i + 1:]]
    if pairs:
        pair_digit = sum(1 for a, b in pairs if a % 10 == b % 10) / len(pairs)
        pair_parity = sum(
            1 for a, b in pairs if a % 2 == b % 2 and a % 10 != b % 10
        ) / len(pairs)
    else:
        pair_digit = pair_parity = 0.0
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": 1.0 if (g >= 3 and g % 5 != 0 and t % g == 0) else 0.0,
        "round_factor_multiple": 1.0 if (g % 5 == 0 and t % g == 0) else 0.0,
        "decade_band_gap": float(max(lo - td, 0, td - hi)),
        "shared_digit_rule": 1.0 if uniform else 0.0,
        "explicit_rule_detected": 1.0 if (len(xs) >= 2 and (uniform or g >= 3)) else 0.0,
        "pair_same_digit": pair_digit,
        "pair_same_parity_other": pair_parity,
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    rule = pm.Data("shared_digit_rule", np.zeros(1, dtype="float64"))
    any_rule = pm.Data("explicit_rule_detected", np.zeros(1, dtype="float64"))
    pair_digit = pm.Data("pair_same_digit", np.zeros(1, dtype="float64"))
    pair_par = pm.Data("pair_same_parity_other", np.zeros(1, dtype="float64"))

    rf_mult = pm.Data("round_factor_multiple", np.zeros(1, dtype="float64"))
    round_boost = pm.Normal("round_boost", mu=0.0, sigma=2.0)
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    preempt = pm.Beta("preempt", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * (1.0 - preempt * rule) * frac_par
    # The examples' mutual similarity on the same metric: the standard.
    mutual_sim = pair_digit + s_parity * pair_par
    # Smooth one-sided shortfall of the target below that standard.
    shortfall = pt.softplus(20.0 * (mutual_sim - mean_sim)) / 20.0

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    standard = pm.HalfNormal("standard", sigma=3.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)
    magnitude_preempt = pm.Beta("magnitude_preempt", alpha=2.0, beta=2.0)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim
            - standard * shortfall
            + factor_boost * cf_mult + round_boost * rf_mult
            + band_boost * (1.0 - magnitude_preempt * any_rule) * band_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
