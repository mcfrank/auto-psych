"""Round-digit kinship in the shortfall-from-mutual-similarity standard.

A refinement of shortfall_from_mutual_similarity. People hold the target to a
standard set by how much the examples resemble one another on the final-digit
metric, and only the target's shortfall below that standard counts against
"yes" (rule-pre-empted parity credit, common-factor boost and rule-gated
decade-band familiarity kept). The one change, taken from
round_digit_kinship_rule_gated, is to the final-digit metric itself: the round
digits 0 and 5 are kin, so a number ending in 0 is partly similar to one
ending in 5 despite their opposite parity, with that kinship credit inferred.
The same metric is used for the target's similarity and for the examples'
mutual similarity.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _round_kin(a, b):
    return a % 5 == 0 and b % 5 == 0 and a % 10 != b % 10


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = float(len(xs))
    same_digit = sum(1 for x in xs if x % 10 == t % 10) / n
    same_parity_other = sum(1 for x in xs if x % 2 == t % 2 and x % 10 != t % 10) / n
    round_kin = sum(1 for x in xs if _round_kin(x, t)) / n
    g = reduce(math.gcd, xs)
    lo = min(x // 10 for x in xs)
    hi = max(x // 10 for x in xs)
    td = t // 10
    uniform = len(xs) >= 2 and len({x % 10 for x in xs}) == 1
    pairs = [(a, b) for i, a in enumerate(xs) for b in xs[i + 1:]]
    if pairs:
        pair_digit = sum(1 for a, b in pairs if a % 10 == b % 10) / len(pairs)
        pair_parity = sum(
            1 for a, b in pairs if a % 2 == b % 2 and a % 10 != b % 10
        ) / len(pairs)
        pair_round = sum(1 for a, b in pairs if _round_kin(a, b)) / len(pairs)
    else:
        pair_digit = pair_parity = pair_round = 0.0
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "frac_round_kin": round_kin,
        "common_factor_multiple": 1.0 if (g >= 3 and t % g == 0) else 0.0,
        "decade_band_gap": float(max(lo - td, 0, td - hi)),
        "shared_digit_rule": 1.0 if uniform else 0.0,
        "explicit_rule_detected": 1.0 if (len(xs) >= 2 and (uniform or g >= 3)) else 0.0,
        "pair_same_digit": pair_digit,
        "pair_same_parity_other": pair_parity,
        "pair_round_kin": pair_round,
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    frac_round = pm.Data("frac_round_kin", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    rule = pm.Data("shared_digit_rule", np.zeros(1, dtype="float64"))
    any_rule = pm.Data("explicit_rule_detected", np.zeros(1, dtype="float64"))
    pair_digit = pm.Data("pair_same_digit", np.zeros(1, dtype="float64"))
    pair_par = pm.Data("pair_same_parity_other", np.zeros(1, dtype="float64"))
    pair_round = pm.Data("pair_round_kin", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    preempt = pm.Beta("preempt", alpha=2.0, beta=2.0)
    # Partial similarity between the round digits 0 and 5.
    s_round = pm.Beta("s_round", alpha=2.0, beta=2.0)
    mean_sim = (
        frac_digit + s_parity * (1.0 - preempt * rule) * frac_par
        + s_round * frac_round
    )
    # The examples' mutual similarity on the same metric: the standard.
    mutual_sim = pair_digit + s_parity * pair_par + s_round * pair_round
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
            + factor_boost * cf_mult
            + band_boost * (1.0 - magnitude_preempt * any_rule) * band_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
