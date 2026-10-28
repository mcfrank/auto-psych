"""Dual-route span exemplar model whose span is marked out by remembered examples.

Refinement of span_filling_size_exemplar: people infer that the program fills
in the stretch of the number line the examples mark out (size mismatch = how
far the target falls outside that stretch), judge final digit, parity and
graded shared factor by recency-weighted similarity (coherence-stretched
attention), consult the "multiples of g" and "ends in d" rules, and relax their
criterion over the session. The single change: the stretch is marked out only
by the examples still remembered. The last-shown example is always remembered;
an example shown `lag` positions earlier is remembered with probability
retention**lag -- the same fading memory that down-weights it as an exemplar --
so the inferred stretch often shrinks toward the most recent example and
targets near the last example are endorsed more than those near the first.
Size similarity is the expectation over which examples are remembered.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
NDIM = 4  # magnitude, final digit, parity, graded shared factor
NSUB = 8  # remembered/forgotten patterns of up to 3 earlier examples
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
    sub_inc = np.ones((n, NSUB, MAXK - 1))
    early_lag = np.zeros((n, MAXK - 1))
    sub_dist = np.zeros((n, NSUB))
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
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        mism[r, 0, :] = 0.0  # size handled by the remembered-span route
        # Earlier examples (all but the last); padded slots have lag 0
        # (remembered with probability 1), so masks forgetting them get 0 mass.
        for j in range(k - 1):
            early_lag[r, j] = k - 1 - j
        for m in range(NSUB):
            bits = [(m >> j) & 1 for j in range(MAXK - 1)]
            sub_inc[r, m] = bits
            kept = [ex[-1]] + [ex[j] for j in range(k - 1) if bits[j]]
            sub_dist[r, m] = max(0, min(kept) - t, t - max(kept)) / 100.0
        ti = row.get("trial_index")
        if ti is not None and str(ti).strip() != "":
            position[r] = min(max(float(ti), 0.0), SESSION_LEN) / SESSION_LEN
        rv = row.get("rating")
        rating[r] = int(float(rv)) if rv is not None and str(rv).strip() != "" else 0
    return {"mismatch": mism, "coherence": coh, "lag": lag, "ex_mask": mask,
            "rule_evidence": rule, "digit_rule_evidence": digit_rule,
            "session_position": position, "subset_included": sub_inc,
            "early_lag": early_lag, "subset_span_dist": sub_dist,
            "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    coherence = pm.Data("coherence", np.zeros((1, NDIM)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    rule_evidence = pm.Data("rule_evidence", np.zeros(1))
    digit_rule_evidence = pm.Data("digit_rule_evidence", np.zeros(1))
    session_position = pm.Data("session_position", np.zeros(1))
    subset_included = pm.Data("subset_included", np.ones((1, NSUB, MAXK - 1)))
    early_lag = pm.Data("early_lag", np.zeros((1, MAXK - 1)))
    subset_span_dist = pm.Data("subset_span_dist", np.zeros((1, NSUB)))

    w_base = pm.LogNormal("w_base", mu=np.log(np.array([0.5, 0.3, 0.5, 0.1])),
                          sigma=1.0)
    tuning = pm.HalfNormal("tuning", sigma=2.0, shape=NDIM)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    rule_weight = pm.HalfNormal("rule_weight", sigma=1.0)
    digit_rule_weight = pm.HalfNormal("digit_rule_weight", sigma=1.0)
    criterion_drift = pm.Normal("criterion_drift", mu=0.0, sigma=1.0)

    log_ret = pt.log(retention)
    attn = w_base[None, :] * pt.exp(tuning[None, :] * coherence)
    dist = pt.sum(attn[:, :, None] * mismatch, axis=1)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * log_ret)
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    other_familiarity = pt.sum(mem * sim, axis=1)

    # The change: probability of each remembered/forgotten pattern of the
    # earlier examples, and the size match to the span the remembered ones mark.
    q = pt.exp(early_lag * log_ret)[:, None, :]
    p_sub = pt.prod(subset_included * q + (1.0 - subset_included) * (1.0 - q), axis=2)
    size_match = pt.sum(p_sub * pt.exp(-attn[:, 0:1] * subset_span_dist), axis=1)
    familiarity = size_match * other_familiarity

    logit = (bias + criterion_drift * session_position + gain * familiarity
             + rule_weight * rule_evidence
             + digit_rule_weight * digit_rule_evidence)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
