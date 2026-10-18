"""Shared-final-digit rule with a criterion that drifts over the session.

A refinement of shared_digit_rule_preempts_parity: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1, same
parity other digit = partial, opposite parity = 0), where an abstracted
"ends in d" rule (two or more examples sharing one final digit) removes an
inferred share of the parity credit, plus a boost for multiples of the
examples' common factor (>= 3) and decade-band familiarity decaying outside
the band. The one change: the response criterion drifts linearly (in log-odds)
with the trial's position in the session, so people become steadily more (or
less) willing to say "yes" as the session goes on.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm

SESSION_LENGTH = 450.0


def _stimulus_features(number_set, target):
    xs = [int(x) for x in str(number_set).split(",")]
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
    uniform = 1.0 if (len(xs) >= 2 and len({x % 10 for x in xs}) == 1) else 0.0
    return same_digit, same_parity_other, divisible, float(band_gap), uniform


def prepare_observed(rows):
    feats = np.array(
        [_stimulus_features(r["number_set"], r["target"]) for r in rows],
        dtype="float64",
    ).reshape(-1, 5)
    # Session position centred on the middle of the session, in units of a session.
    pos = np.array([float(r["trial_index"]) for r in rows], dtype="float64")
    pos = pos / SESSION_LENGTH - 0.5
    return {
        "frac_same_digit": feats[:, 0],
        "frac_same_parity_other": feats[:, 1],
        "common_factor_multiple": feats[:, 2],
        "decade_band_gap": feats[:, 3],
        "shared_digit_rule": feats[:, 4],
        "session_position": pos,
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    rule = pm.Data("shared_digit_rule", np.zeros(1, dtype="float64"))
    session_pos = pm.Data("session_position", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    # Share of parity credit removed once the "ends in d" rule is abstracted.
    preempt = pm.Beta("preempt", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * (1.0 - preempt * rule) * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)
    # Change in the yes-criterion (log-odds) from session start to end.
    criterion_drift = pm.Normal("criterion_drift", mu=0.0, sigma=1.0)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + factor_boost * cf_mult
            + band_boost * band_sim + criterion_drift * session_pos
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
