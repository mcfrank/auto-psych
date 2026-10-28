"""Dimension-specific coherence exemplar plus an abstracted shared-factor rule.

Refinement of dimension_specific_coherence_exemplar: people judge a target by
its recency-weighted exponential similarity to the remembered examples (size,
final digit, parity, graded shared factor), with each dimension's attention
stretched by its own coherence sensitivity. The single change: when all
examples share a common factor g >= 3, people also abstract the rule
"multiples of g" and consult it at the decision as evidence separate from
similarity, pushing "yes" up for multiples and down for non-multiples
irrespective of size distance, with a strength proportional to how unlikely
the shared factor is by coincidence, (k - 1) * log(g).
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
            g = reduce(math.gcd, ex) if k >= 2 else 0
            cache[s] = (c, g)
        c, g = cache[s]
        coh[r] = c
        if g >= 3:
            # Signed rule evidence, scaled by how surprising the shared factor is.
            rule[r] = (k - 1) * math.log(g) * (1.0 if t % g == 0 else -1.0)
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
    # How strongly inter-example coherence stretches attention, per dimension.
    tuning = pm.HalfNormal("tuning", sigma=2.0, shape=NDIM)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # Weight of the abstracted "multiples of g" rule at the decision (the change).
    rule_weight = pm.HalfNormal("rule_weight", sigma=1.0)

    attn = w_base[None, :] * pt.exp(tuning[None, :] * coherence)          # (n, D)
    dist = pt.sum(attn[:, :, None] * mismatch, axis=1)           # (n, K)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    logit = bias + gain * familiarity + rule_weight * rule_evidence
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
