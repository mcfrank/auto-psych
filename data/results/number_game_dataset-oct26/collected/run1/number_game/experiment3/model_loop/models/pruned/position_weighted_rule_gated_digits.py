"""Serial-position-weighted digit comparison inside rule-gated decade familiarity.

People judge the target as rule_gated_decade_familiarity says: final-digit
similarity to the examples (same digit = 1, same parity other digit = partial,
opposite parity = 0), with an abstracted "ends in d" rule removing an inferred
share of parity credit, a boost for multiples of the examples' common factor
(>= 3) and decade-band familiarity cut by an inferred share when an explicit
rule is detected. The one change: examples are not held equally in memory --
each example's weight in the final-digit comparison changes steadily with its
serial position (recency or primacy, direction and strength inferred), so the
digit similarity is a position-weighted rather than plain average.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    g = reduce(math.gcd, xs)
    divisible = 1.0 if (g >= 3 and t % g == 0) else 0.0
    lo = min(x // 10 for x in xs)
    hi = max(x // 10 for x in xs)
    td = t // 10
    band_gap = max(lo - td, 0, td - hi)
    uniform = len(xs) >= 2 and len({x % 10 for x in xs}) == 1
    rule_found = len(xs) >= 2 and (uniform or g >= 3)
    out = {
        "common_factor_multiple": divisible,
        "decade_band_gap": float(band_gap),
        "shared_digit_rule": 1.0 if uniform else 0.0,
        "explicit_rule_detected": 1.0 if rule_found else 0.0,
    }
    rev = xs[-K:][::-1]  # lag 0 = last-shown example
    for k in range(K):
        if k < len(rev):
            x = rev[k]
            same = 1.0 if x % 10 == t % 10 else 0.0
            par = 1.0 if (x % 2 == t % 2 and same == 0.0) else 0.0
            m = 1.0
        else:
            same, par, m = 0.0, 0.0, 0.0
        out[f"pw_same_lag{k}"] = same
        out[f"pw_par_lag{k}"] = par
        out[f"pw_mask_lag{k}"] = m
    return out


with pm.Model() as model:
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    rule = pm.Data("shared_digit_rule", np.zeros(1, dtype="float64"))
    any_rule = pm.Data("explicit_rule_detected", np.zeros(1, dtype="float64"))
    same = [pm.Data(f"pw_same_lag{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    par = [pm.Data(f"pw_par_lag{k}", np.zeros(1, dtype="float64")) for k in range(K)]
    mask = [pm.Data(f"pw_mask_lag{k}", np.ones(1, dtype="float64")) for k in range(K)]

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    preempt = pm.Beta("preempt", alpha=2.0, beta=2.0)
    # Log-weight change per position back from the last-shown example
    # (positive = recency, negative = primacy).
    decay = pm.Normal("decay", mu=0.0, sigma=1.0)

    par_credit = s_parity * (1.0 - preempt * rule)
    total = 0.0
    wsum = 1e-9
    for k in range(K):
        w = mask[k] * pm.math.exp(-decay * k)
        total = total + w * (same[k] + par_credit * par[k])
        wsum = wsum + w
    mean_sim = total / wsum

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)
    magnitude_preempt = pm.Beta("magnitude_preempt", alpha=2.0, beta=2.0)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + factor_boost * cf_mult
            + band_boost * (1.0 - magnitude_preempt * any_rule) * band_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
