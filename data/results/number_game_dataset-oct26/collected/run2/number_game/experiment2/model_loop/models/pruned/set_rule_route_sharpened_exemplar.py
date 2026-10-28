"""Coherence-sharpened exemplar similarity plus an abstracted set-rule route.

Refinement of coherence_sharpened_decision_exemplar: people judge a target by
its recency-weighted exponential similarity to the remembered examples (size,
final digit, parity, graded shared factor), with attention stretched toward the
dimensions on which the examples resemble one another and set coherence
sharpening the familiarity-to-answer gain. The single change: when all examples
share a final digit d or a common factor g >= 3, people abstract the narrowest
such rule ("ends in d" or "multiples of g") and consult it at the decision as
evidence separate from similarity, pushing "yes" up for targets obeying it and
down for those violating it irrespective of size distance, with a strength equal
to how unlikely the coincidence is under the size principle:
(k - 1) * log(100 / |rule|).
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
NDIM = 4  # magnitude, final digit, parity, graded shared factor


def _factor_mismatch(t, x):
    m = min(t, x)
    if m <= 1:
        return 0.0
    return 1.0 - math.log(math.gcd(t, x)) / math.log(m)


def _set_rule(ex):
    """Narrowest shared rule of the set as a membership test and its size."""
    k = len(ex)
    if k < 2:
        return None
    cands = []
    if len({x % 10 for x in ex}) == 1:
        d = ex[0] % 10
        size = sum(1 for n in range(1, 101) if n % 10 == d)
        cands.append((size, lambda t, d=d: t % 10 == d))
    g = reduce(math.gcd, ex)
    if g >= 3:
        cands.append((100 // g, lambda t, g=g: t % g == 0))
    if not cands:
        return None
    size, test = min(cands, key=lambda c: c[0])
    return test, (k - 1) * math.log(100.0 / size)


def prepare_observed(rows):
    n = len(rows)
    mism = np.ones((n, NDIM, MAXK))
    coh = np.zeros((n, NDIM))
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    rule = np.zeros(n)
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
            cache[s] = (c, _set_rule(ex))
        c, sr = cache[s]
        coh[r] = c
        if sr is not None:
            test, strength = sr
            rule[r] = strength * (1.0 if test(t) else -1.0)
        for i, x in enumerate(ex):
            mism[r, 0, i] = abs(t - x) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        rating[r] = int(float(row["rating"]))
    return {"mismatch": mism, "coherence": coh, "lag": lag, "ex_mask": mask,
            "rule_evidence": rule, "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    coherence = pm.Data("coherence", np.zeros((1, NDIM)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    rule_evidence = pm.Data("rule_evidence", np.zeros(1))

    # Baseline attention on each dimension (magnitude is on a 0-1 scale).
    w_base = pm.HalfNormal("w_base", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    # How strongly inter-example coherence on a dimension stretches attention.
    tuning = pm.HalfNormal("tuning", sigma=2.0)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # How strongly set coherence sharpens the familiarity-to-answer mapping.
    sharpen = pm.Normal("sharpen", 0.0, 1.0)
    # Weight of the abstracted set rule at the decision (the change).
    rule_weight = pm.HalfNormal("rule_weight", sigma=1.0)

    attn = w_base[None, :] * pt.exp(tuning * coherence)          # (n, D)
    dist = pt.sum(attn[:, :, None] * mismatch, axis=1)           # (n, K)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    decisiveness = gain * pt.exp(sharpen * pt.mean(coherence, axis=1))
    logit = bias + decisiveness * familiarity + rule_weight * rule_evidence
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
