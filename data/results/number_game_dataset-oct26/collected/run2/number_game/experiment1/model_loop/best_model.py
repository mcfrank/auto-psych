"""Graded-factor coherence-tuned exemplar similarity with a coherence-sharpened decision.

Refinement of graded_factor_coherence_tuned_exemplar: people judge a target by
its recency-weighted exponential similarity to the remembered examples (size,
final digit, parity, graded shared factor), with attention stretched toward the
dimensions on which the examples resemble one another. The single change: the
examples' overall coherence (their mean resemblance across dimensions) also
sharpens the decision rule, scaling the gain that maps familiarity to "yes" by
exp(sharpen * mean coherence). After a coherent set people answer almost
deterministically from similarity; after an incoherent set or a single example
(coherence 0, baseline gain) their answers stay noisy.
"""
import math

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
            cache[s] = c
        coh[r] = cache[s]
        for i, x in enumerate(ex):
            mism[r, 0, i] = abs(t - x) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        rating[r] = int(float(row["rating"]))
    return {"mismatch": mism, "coherence": coh, "lag": lag, "ex_mask": mask,
            "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    coherence = pm.Data("coherence", np.zeros((1, NDIM)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))

    # Baseline attention on each dimension (magnitude is on a 0-1 scale).
    w_base = pm.HalfNormal("w_base", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    # How strongly inter-example coherence on a dimension stretches attention.
    tuning = pm.HalfNormal("tuning", sigma=2.0)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # How strongly set coherence sharpens the familiarity-to-answer mapping.
    sharpen = pm.Normal("sharpen", 0.0, 1.0)

    attn = w_base[None, :] * pt.exp(tuning * coherence)          # (n, D)
    dist = pt.sum(attn[:, :, None] * mismatch, axis=1)           # (n, K)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    decisiveness = gain * pt.exp(sharpen * pt.mean(coherence, axis=1))
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + decisiveness * familiarity), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
