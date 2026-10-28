"""Dual-route disengaging exemplar model with upward-biased size similarity.

Refinement of iter1_candidate4 (itself a refinement of
dual_route_factor_and_digit_rule_exemplar). The single change here: size
distance is directional -- a target larger than an example is treated as closer
than an equally distant smaller one (attention to size scaled by exp(-asym) for
upward and exp(+asym) for downward differences), so people generalise upward
more than downward. Otherwise as before: people judge a target
by its recency-weighted exponential similarity to the remembered examples
(size, final digit, parity, graded shared factor; attention stretched by
coherence) and consult abstracted "multiples of g" and "ends in d" rules at the
decision. The single change: on a share of trials that grows over the session,
people disengage and guess with their own yes-bias instead of consulting the
examples, so late answers drift toward that guessing rate for every target
alike. When a stimulus's session position is unknown the guessing share is
averaged over the session (the yes-probability is linear in it, so this is the
exact marginal).
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
NDIM = 4  # magnitude, final digit, parity, graded shared factor
SESSION_LEN = 450.0
GRID = np.linspace(0.0, 1.0, 11)


def _factor_mismatch(t, x):
    m = min(t, x)
    if m <= 1:
        return 0.0
    return 1.0 - math.log(math.gcd(t, x)) / math.log(m)


def prepare_observed(rows):
    n = len(rows)
    mism = np.ones((n, NDIM, MAXK))
    coh = np.zeros((n, NDIM))
    lag = np.zeros((n, MAXK))
    direction = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    rule = np.zeros(n)
    digit_rule = np.zeros(n)
    pos = np.zeros(n)
    pos_known = np.zeros(n)
    rating = np.zeros(n, dtype="int64")
    cache = {}
    for r, row in enumerate(rows):
        s = str(row["number_set"])
        ex = [int(x) for x in s.split(",")][:MAXK]
        t = int(row["target"])
        k = len(ex)
        if s not in cache:
            pairs = [(ex[i], ex[j]) for i in range(k) for j in range(i + 1, k)]
            c = np.zeros(NDIM)
            if pairs:
                c[0] = np.mean([max(0.0, 1.0 - abs(a - b) / 50.0) for a, b in pairs])
                c[1] = np.mean([a % 10 == b % 10 for a, b in pairs])
                c[2] = np.mean([a % 2 == b % 2 for a, b in pairs])
                c[3] = np.mean([1.0 - _factor_mismatch(a, b) for a, b in pairs])
            g = reduce(math.gcd, ex) if k >= 2 else 0
            d = ex[0] % 10 if k >= 2 and len({x % 10 for x in ex}) == 1 else -1
            cache[s] = (c, g, d)
        c, g, d = cache[s]
        if d >= 0:
            digit_rule[r] = (k - 1) * math.log(10.0) * (1.0 if t % 10 == d else -1.0)
        coh[r] = c
        if g >= 3:
            rule[r] = (k - 1) * math.log(g) * (1.0 if t % g == 0 else -1.0)
        for i, x in enumerate(ex):
            mism[r, 0, i] = abs(t - x) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            direction[r, i] = float(np.sign(t - x))
            mask[r, i] = 1.0
        # Session position in [0, 1]; a bare stimulus has none (marginalised).
        ti = row.get("trial_index")
        if ti is not None and str(ti).strip() != "":
            pos[r] = min(max(float(ti) / SESSION_LEN, 0.0), 1.0)
            pos_known[r] = 1.0
        rating[r] = int(float(row.get("rating", 0) or 0))
    return {"mismatch": mism, "coherence": coh, "lag": lag, "direction": direction, "ex_mask": mask,
            "rule_evidence": rule, "digit_rule_evidence": digit_rule,
            "session_pos": pos, "pos_known": pos_known, "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    coherence = pm.Data("coherence", np.zeros((1, NDIM)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    direction = pm.Data("direction", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    rule_evidence = pm.Data("rule_evidence", np.zeros(1))
    digit_rule_evidence = pm.Data("digit_rule_evidence", np.zeros(1))
    session_pos = pm.Data("session_pos", np.zeros(1))
    pos_known = pm.Data("pos_known", np.zeros(1))

    w_base = pm.HalfNormal("w_base", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    tuning = pm.HalfNormal("tuning", sigma=2.0, shape=NDIM)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    rule_weight = pm.HalfNormal("rule_weight", sigma=1.0)
    digit_rule_weight = pm.HalfNormal("digit_rule_weight", sigma=1.0)

    # Disengagement share rises (logistically) over the session,
    # and disengaged answers are guesses with their own yes-rate.
    lapse_start = pm.Normal("lapse_start", mu=-2.5, sigma=1.0)
    lapse_growth = pm.Normal("lapse_growth", mu=0.0, sigma=2.0)
    guess_yes = pm.Beta("guess_yes", alpha=2.0, beta=2.0)

    attn = w_base[None, :] * pt.exp(tuning[None, :] * coherence)
    # The change: upward size differences count for less than downward ones.
    size_asym = pm.Normal("size_asym", mu=0.0, sigma=0.5)
    dir_scale = pt.exp(-size_asym * direction)                    # (n, K)
    dist = (attn[:, 0, None] * mismatch[:, 0, :] * dir_scale
            + pt.sum(attn[:, 1:, None] * mismatch[:, 1:, :], axis=1))
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    logit = (bias + gain * familiarity + rule_weight * rule_evidence
             + digit_rule_weight * digit_rule_evidence)
    p_engaged = pm.math.sigmoid(logit)

    lapse_known = pm.math.sigmoid(lapse_start + lapse_growth * session_pos)
    lapse_avg = pt.mean(pm.math.sigmoid(lapse_start + lapse_growth * GRID))
    lapse = pos_known * lapse_known + (1.0 - pos_known) * lapse_avg

    p = (1.0 - lapse) * p_engaged + lapse * guess_yes
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
