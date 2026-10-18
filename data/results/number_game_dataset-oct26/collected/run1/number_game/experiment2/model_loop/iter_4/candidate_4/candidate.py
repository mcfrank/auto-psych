"""Session-long criterion relaxation on top of rule-gated decade familiarity.

A refinement of rule_gated_decade_familiarity: "yes" still grows with the
target's average final-digit similarity to the examples (an abstracted
"ends in d" rule removes an inferred share of parity credit), a boost for
multiples of the examples' common factor (>= 3), and decade-band familiarity
that is cut by an inferred share when an explicit rule is detected. The one
change: the response criterion relaxes with experience over the session --
people come to believe the program produces more numbers -- so the bias
toward "yes" rises with trial position and saturates, with inferred size and
time constant.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm


def _features(number_set, target):
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
    return (same_digit, same_parity_other, divisible, float(band_gap),
            1.0 if uniform else 0.0, 1.0 if rule_found else 0.0)


def prepare_observed(rows):
    cache = {}
    feats = []
    for r in rows:
        key = (r["number_set"], r["target"])
        if key not in cache:
            cache[key] = _features(*key)
        feats.append(cache[key])
    f = np.asarray(feats, dtype="float64")
    return {
        "frac_same_digit": f[:, 0],
        "frac_same_parity_other": f[:, 1],
        "common_factor_multiple": f[:, 2],
        "decade_band_gap": f[:, 3],
        "shared_digit_rule": f[:, 4],
        "explicit_rule_detected": f[:, 5],
        # Session position in units of 100 trials.
        "session_position": np.asarray(
            [float(r["trial_index"]) / 100.0 for r in rows], dtype="float64"),
        "rating": np.asarray([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    rule = pm.Data("shared_digit_rule", np.zeros(1, dtype="float64"))
    any_rule = pm.Data("explicit_rule_detected", np.zeros(1, dtype="float64"))
    session = pm.Data("session_position", np.zeros(1, dtype="float64"))

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

    # Criterion relaxation: total shift in log-odds reached late in the
    # session, approached with time constant tau (in 100s of trials).
    relax = pm.Normal("relax", mu=0.0, sigma=1.0)
    tau = pm.LogNormal("tau", mu=np.log(1.5), sigma=0.75)
    criterion_shift = relax * (1.0 - pm.math.exp(-session / tau))

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + gain * mean_sim + factor_boost * cf_mult
            + band_boost * (1.0 - magnitude_preempt * any_rule) * band_sim
            + criterion_shift
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
