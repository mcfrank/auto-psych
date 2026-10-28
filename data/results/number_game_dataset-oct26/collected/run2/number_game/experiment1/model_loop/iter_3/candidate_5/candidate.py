"""Dimension-specific coherence-tuned attention exemplar model.

Refinement of coherence_tuned_attention_exemplar: people judge a target by its
recency-weighted similarity to the remembered examples, with attention
stretched toward the dimensions (size, final digit, parity, shared factor) on
which the examples resemble one another, but each kind of shared property has
its own coherence sensitivity (one stretch rate per dimension instead of one
common rate): a run of examples ending in the same digit captures attention
far more readily than shared parity. A single example leaves attention at
baseline.
"""
import math
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
NDIM = 4  # magnitude, final digit, parity, shared factor


def _factor_match(a, b):
    return a % b == 0 or b % a == 0 or math.gcd(a, b) >= 3


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
                c[3] = np.mean([_factor_match(a, b) for a, b in pairs])
            cache[s] = c
        coh[r] = cache[s]
        for i, x in enumerate(ex):
            mism[r, 0, i] = abs(t - x) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = 0.0 if _factor_match(t, x) else 1.0
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
    # Per-dimension sensitivity of attention to inter-example coherence.
    tuning = pm.HalfNormal("tuning", sigma=3.0, shape=NDIM)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    attn = w_base[None, :] * pt.exp(tuning[None, :] * coherence)  # (n, D)
    dist = pt.sum(attn[:, :, None] * mismatch, axis=1)            # (n, K)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + gain * familiarity), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
