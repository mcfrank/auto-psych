"""Contains-digit concepts with a response criterion that drifts over the session.

A refinement of contains_digit_concept: final-digit similarity judged against the
examples' mutual similarity, rule-pre-empted parity credit, a common-factor
boost, rule-gated decade-band familiarity and "contains the digit d" support are
all kept. The one change: people's criterion for saying "yes" shifts steadily
with how far into the session the trial comes (direction and rate inferred), so
the same (set, target) is accepted more or less often late than early.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm

MAX_TRIAL = 450.0


def _stimulus_features(number_set, target):
    xs = [int(x) for x in str(number_set).split(",")]
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
    tdig = set(str(t))
    contains_shared = sum(
        1 for x in xs if (tdig & set(str(x))) and x % 10 != t % 10
    ) / n
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": 1.0 if (g >= 3 and t % g == 0) else 0.0,
        "decade_band_gap": float(max(lo - td, 0, td - hi)),
        "shared_digit_rule": 1.0 if uniform else 0.0,
        "explicit_rule_detected": 1.0 if (len(xs) >= 2 and (uniform or g >= 3)) else 0.0,
        "pair_same_digit": pair_digit,
        "pair_same_parity_other": pair_parity,
        "frac_contains_digit_other": contains_shared,
    }


def prepare_observed(rows):
    cache = {}
    cols = {}
    for r in rows:
        key = (str(r["number_set"]), str(r["target"]))
        if key not in cache:
            cache[key] = _stimulus_features(*key)
        for k, v in cache[key].items():
            cols.setdefault(k, []).append(v)
        # Session position, centred: -0.5 at the start, +0.5 at the end.
        cols.setdefault("session_position", []).append(
            float(r["trial_index"]) / MAX_TRIAL - 0.5
        )
        cols.setdefault("rating", []).append(int(float(r["rating"])))
    out = {k: np.asarray(v, dtype="float64") for k, v in cols.items()}
    out["rating"] = out["rating"].astype("int64")
    return out


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
    band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
    rule = pm.Data("shared_digit_rule", np.zeros(1, dtype="float64"))
    any_rule = pm.Data("explicit_rule_detected", np.zeros(1, dtype="float64"))
    pair_digit = pm.Data("pair_same_digit", np.zeros(1, dtype="float64"))
    pair_par = pm.Data("pair_same_parity_other", np.zeros(1, dtype="float64"))
    contains = pm.Data("frac_contains_digit_other", np.zeros(1, dtype="float64"))
    session_pos = pm.Data("session_position", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    preempt = pm.Beta("preempt", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * (1.0 - preempt * rule) * frac_par
    mutual_sim = pair_digit + s_parity * pair_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    standard = pm.HalfNormal("standard", sigma=3.0)
    factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0)
    band_boost = pm.HalfNormal("band_boost", sigma=2.0)
    band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
    band_sim = pm.math.exp(-band_gap / band_width)
    magnitude_preempt = pm.Beta("magnitude_preempt", alpha=2.0, beta=2.0)
    contains_boost = pm.Normal("contains_boost", mu=0.0, sigma=1.0)
    # Session-wide drift of the "yes" criterion (log-odds over the whole session).
    criterion_drift = pm.Normal("criterion_drift", mu=0.0, sigma=1.0)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(
            intercept + criterion_drift * session_pos
            + gain * mean_sim
            - standard * mutual_sim * (1.0 - frac_digit)
            + factor_boost * cf_mult
            + band_boost * (1.0 - magnitude_preempt * any_rule) * band_sim
            + contains_boost * contains
        ),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
