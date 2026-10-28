"""Recency-anchored span exemplar model.

Refinement of span_filling_size_exemplar: people judge a target by its
recency-weighted exponential similarity to the remembered examples (final
digit, parity, graded shared factor; attention stretched by coherence), consult
the "multiples of g" and "ends in d" rules as separate evidence, relax their
criterion over the session, and judge size by how far the target falls outside
a filled-in span of the number line. The single change: the span is anchored on
the most recently shown example, and earlier examples' positions fade in
memory, so the stretch each earlier example adds to the span shrinks back toward
the last example by a factor per position back. Targets on the last example's
side of the set fall inside the remembered span more often than equally placed
targets on the first example's side (recency in size generalization).
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
NDIM = 4  # magnitude, final digit, parity, graded shared factor
SESSION_LEN = 450.0


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
    mask = np.zeros((n, MAXK))
    rule = np.zeros(n)
    digit_rule = np.zeros(n)
    position = np.zeros(n)
    ex_pos = np.zeros((n, MAXK))
    tgt_pos = np.zeros(n)
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
            mism[r, 0, i] = 0.0  # size handled by the remembered span below
            ex_pos[r, i] = x / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        for i in range(k, MAXK):
            ex_pos[r, i] = ex[-1] / 100.0  # padded slots sit on the anchor
        tgt_pos[r] = t / 100.0
        ti = row.get("trial_index")
        if ti is not None and str(ti).strip() != "":
            position[r] = min(max(float(ti), 0.0), SESSION_LEN) / SESSION_LEN
        rv = row.get("rating")
        rating[r] = int(float(rv)) if rv is not None and str(rv).strip() != "" else 0
    return {"mismatch": mism, "coherence": coh, "lag": lag, "ex_mask": mask,
            "rule_evidence": rule, "digit_rule_evidence": digit_rule,
            "session_position": position, "ex_pos": ex_pos,
            "tgt_pos": tgt_pos, "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    coherence = pm.Data("coherence", np.zeros((1, NDIM)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    rule_evidence = pm.Data("rule_evidence", np.zeros(1))
    digit_rule_evidence = pm.Data("digit_rule_evidence", np.zeros(1))
    session_position = pm.Data("session_position", np.zeros(1))
    ex_pos = pm.Data("ex_pos", np.zeros((1, MAXK)))
    tgt_pos = pm.Data("tgt_pos", np.zeros(1))

    w_base = pm.HalfNormal("w_base", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    tuning = pm.HalfNormal("tuning", sigma=2.0, shape=NDIM)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    rule_weight = pm.HalfNormal("rule_weight", sigma=1.0)
    digit_rule_weight = pm.HalfNormal("digit_rule_weight", sigma=1.0)
    # How far the yes-criterion relaxes from session start to end (the change).
    criterion_drift = pm.Normal("criterion_drift", mu=0.0, sigma=1.0)

    # The change: how much of each earlier example's reach survives in memory.
    span_retention = pm.Beta("span_retention", alpha=2.0, beta=2.0)
    anchor = pt.sum(ex_pos * pt.eq(lag, 0) * ex_mask, axis=1, keepdims=True)
    reach = pt.exp(lag * pt.log(span_retention))
    ends = anchor + reach * (ex_pos - anchor)
    lo = pt.min(ends, axis=1)
    hi = pt.max(ends, axis=1)
    outside = pt.maximum(lo - tgt_pos, 0.0) + pt.maximum(tgt_pos - hi, 0.0)
    mismatch = pt.set_subtensor(
        mismatch[:, 0, :], outside[:, None] * pt.ones_like(ex_mask))

    attn = w_base[None, :] * pt.exp(tuning[None, :] * coherence)
    dist = pt.sum(attn[:, :, None] * mismatch, axis=1)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    logit = (bias + criterion_drift * session_position + gain * familiarity
             + rule_weight * rule_evidence
             + digit_rule_weight * digit_rule_evidence)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
