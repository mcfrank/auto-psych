"""Final-digit-rule parity suppression with a response criterion that loosens over the session.

A refinement of final_digit_rule_suppresses_parity: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1, same
parity other digit = partial credit shrinking with how consistently the
examples share one final digit, opposite parity = 0), plus a boost for
multiples of the examples' common factor (>= 3) and decade-band familiarity.
The one change: the yes/no criterion drifts linearly with the trial's position
in the session, so people grow more inclined to say "yes" as they go on.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm


def _stimulus_features(number_set, target):
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
    }


def prepare_observed(rows):
    feats = [_stimulus_features(r["number_set"], r["target"]) for r in rows]
    out = {k: np.array([f[k] for f in feats], dtype="float64") for k in feats[0]}
    # Session position, centred and scaled to roughly [-1, 1] over 450 trials.
    out["session_position"] = np.array(
        [(float(r["trial_index"]) - 225.0) / 225.0 for r in rows], dtype="float64"
    )
    out["rating"] = np.array([int(float(r["rating"])) for r in rows], dtype="int64")
    return out


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    conc = pm.Data("final_digit_concentration", np.zeros(1, dtype="float64"))
    session_pos = pm.Data("session_position", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    rule_suppression = pm.Beta("rule_suppression", alpha=1.5, beta=1.5)
    parity_credit = s_parity * (1.0 - rule_suppression * conc)
    mean_sim = frac_digit + parity_credit * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)
    # Criterion drift: change in yes-bias (log-odds) per half-session.
    criterion_drift = pm.Normal("criterion_drift", mu=0.0, sigma=1.0)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + criterion_drift * session_pos + gain * mean_sim
            + factor_boost * cf_mult + band_boost * band_sim
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
