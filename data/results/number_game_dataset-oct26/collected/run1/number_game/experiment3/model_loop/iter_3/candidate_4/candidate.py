"""Digit-rule size principle on top of privileged named number concepts.

A refinement of privileged_named_number_concepts (shortfall from the examples'
mutual final-digit similarity, rule-pre-empted parity credit, rule-gated
decade-band familiarity, the privileged "even numbers" rule and separate
round/non-round common-factor boosts, all kept). The one change: a size
principle on the shared-final-digit rule -- when all examples end in the same
digit, a target ending in a different digit pays an inferred extra penalty for
each confirming example beyond the first, so rule violators are rejected more
firmly as consistent examples accumulate.
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
    pairs = [(a, b) for i, a in enumerate(xs) for b in xs[i + 1:]]
    if pairs:
        pair_digit = sum(1 for a, b in pairs if a % 10 == b % 10) / len(pairs)
        pair_parity = sum(
            1 for a, b in pairs if a % 2 == b % 2 and a % 10 != b % 10
        ) / len(pairs)
    else:
        pair_digit = pair_parity = 0.0
    all_even = all(x % 2 == 0 for x in xs)
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
        # The target violates an "even numbers" rule the all-even set evokes.
        "breaks_even_rule": 1.0 if (all_even and t % 2 == 1) else 0.0,
        # Confirming examples beyond the first for a violated shared-digit rule.
        "digit_rule_violation_evidence": (
            (n - 1.0) if (uniform and t % 10 != xs[0] % 10) else 0.0
        ),
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
    round_mult = pm.Data("round_factor_multiple", np.zeros(1, dtype="float64"))
    breaks_even = pm.Data("breaks_even_rule", np.zeros(1, dtype="float64"))
    rule_evidence = pm.Data("digit_rule_violation_evidence", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    preempt = pm.Beta("preempt", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * (1.0 - preempt * rule) * frac_par
    mutual_sim = pair_digit + s_parity * pair_par
    shortfall = pt.softplus(20.0 * (mutual_sim - mean_sim)) / 20.0

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    standard = pm.HalfNormal("standard", sigma=3.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    round_boost = pm.Normal("round_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)
    magnitude_preempt = pm.Beta("magnitude_preempt", alpha=2.0, beta=2.0)
    # Extra rejection of odd targets once all examples are even.
    even_rule_penalty = pm.HalfNormal("even_rule_penalty", sigma=2.0)
    # Size principle: extra rejection per confirming example of a broken digit rule.
    digit_rule_size = pm.HalfNormal("digit_rule_size", sigma=1.0)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim
            - standard * shortfall
            + factor_boost * cf_mult + round_boost * round_mult
            + band_boost * (1.0 - magnitude_preempt * any_rule) * band_sim
            - even_rule_penalty * breaks_even
            - digit_rule_size * rule_evidence
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
