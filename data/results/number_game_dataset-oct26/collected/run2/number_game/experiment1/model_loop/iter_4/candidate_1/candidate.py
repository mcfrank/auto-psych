"""Coherence-sharpened exemplar similarity on a compressed mental number line.

Refinement of coherence_sharpened_decision_exemplar: people judge a target by
its recency-weighted exponential similarity to the remembered examples (size,
final digit, parity, graded shared factor), with attention stretched toward the
dimensions on which the examples resemble one another and set coherence
sharpening the decision rule. The single change: size closeness is judged on a
compressed (Weber-like) number line f(n) = (n^gamma - 1) / (100^gamma - 1),
with the compression gamma inferred (gamma -> 0 is logarithmic, 1 is linear),
so equal differences among small numbers feel larger than among large ones.
Size coherence among the examples is measured on the same compressed line.
"""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
MAXP = 6
NDIM = 4  # magnitude, final digit, parity, graded shared factor


def _factor_mismatch(t, x):
    m = min(t, x)
    if m <= 1:
        return 0.0
    return 1.0 - math.log(math.gcd(t, x)) / math.log(m)


def prepare_observed(rows):
    n = len(rows)
    mism = np.ones((n, NDIM - 1, MAXK))  # final digit, parity, factor
    coh = np.zeros((n, NDIM - 1))
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    ex_val = np.ones((n, MAXK))
    tgt = np.ones(n)
    pa = np.ones((n, MAXP))
    pb = np.ones((n, MAXP))
    pmask = np.zeros((n, MAXP))
    rating = np.zeros(n, dtype="int64")
    cache = {}
    for r, row in enumerate(rows):
        s = str(row["number_set"])
        ex = [int(x) for x in s.split(",")][:MAXK]
        t = int(row["target"])
        k = len(ex)
        pairs = [(ex[i], ex[j]) for i in range(k) for j in range(i + 1, k)]
        if s not in cache:
            c = np.zeros(NDIM - 1)
            if pairs:
                c[0] = np.mean([a % 10 == b % 10 for a, b in pairs])
                c[1] = np.mean([a % 2 == b % 2 for a, b in pairs])
                c[2] = np.mean([1.0 - _factor_mismatch(a, b) for a, b in pairs])
            cache[s] = c
        coh[r] = cache[s]
        for p, (a, b) in enumerate(pairs):
            pa[r, p], pb[r, p], pmask[r, p] = a, b, 1.0
        for i, x in enumerate(ex):
            ex_val[r, i] = x
            mism[r, 0, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 1, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 2, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        tgt[r] = t
        rating[r] = int(float(row["rating"]))
    return {"mismatch_cat": mism, "coherence_cat": coh, "lag": lag,
            "ex_mask": mask, "ex_val": ex_val, "target_val": tgt,
            "pair_a": pa, "pair_b": pb, "pair_mask": pmask, "rating": rating}


with pm.Model() as model:
    mismatch_cat = pm.Data("mismatch_cat", np.zeros((1, NDIM - 1, MAXK)))
    coherence_cat = pm.Data("coherence_cat", np.zeros((1, NDIM - 1)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    ex_val = pm.Data("ex_val", np.ones((1, MAXK)))
    target_val = pm.Data("target_val", np.ones(1))
    pair_a = pm.Data("pair_a", np.ones((1, MAXP)))
    pair_b = pm.Data("pair_b", np.ones((1, MAXP)))
    pair_mask = pm.Data("pair_mask", np.zeros((1, MAXP)))

    # Compression of the mental number line (small = logarithmic, 1 = linear).
    gamma = pm.LogNormal("gamma", mu=math.log(0.5), sigma=0.7)

    def line(v):
        return (pt.power(v, gamma) - 1.0) / (pt.power(100.0, gamma) - 1.0)

    mag_mis = pt.abs(line(target_val)[:, None] - line(ex_val))          # (n, K)
    pair_close = pt.maximum(0.0, 1.0 - 2.0 * pt.abs(line(pair_a) - line(pair_b)))
    npairs = pt.sum(pair_mask, axis=1)
    mag_coh = pt.sum(pair_mask * pair_close, axis=1) / pt.maximum(npairs, 1.0)
    coherence = pt.concatenate([mag_coh[:, None], coherence_cat], axis=1)  # (n, D)
    mismatch = pt.concatenate([mag_mis[:, None, :], mismatch_cat], axis=1)  # (n, D, K)

    w_base = pm.HalfNormal("w_base", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    tuning = pm.HalfNormal("tuning", sigma=2.0)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    sharpen = pm.Normal("sharpen", 0.0, 1.0)

    attn = w_base[None, :] * pt.exp(tuning * coherence)
    dist = pt.sum(attn[:, :, None] * mismatch, axis=1)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    decisiveness = gain * pt.exp(sharpen * pt.mean(coherence, axis=1))
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + decisiveness * familiarity), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
