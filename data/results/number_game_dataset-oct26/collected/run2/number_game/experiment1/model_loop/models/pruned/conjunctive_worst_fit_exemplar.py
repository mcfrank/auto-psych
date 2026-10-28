"""Conjunctive (worst-fit) exemplar generalization.

People judge a target by how well it fits the examples as a whole,
conjunctively: rather than summing similarity to each remembered example, they
require the target to fit every example, so familiarity is set by its
worst-fitting examples -- a recency-weighted soft maximum of its psychological
distances to the examples (size, final digit, parity, graded shared factor).
A strictness parameter moves from averaging distances (near 0) to the single
worst-fitting example (large). Targets resembling one example but lacking the
property all examples share are rejected, sharpening "multiples of g" and
"ends in d" generalization on multi-example sets.
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
    mism = np.zeros((n, NDIM, MAXK))
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")][:MAXK]
        t = int(row["target"])
        k = len(ex)
        for i, x in enumerate(ex):
            mism[r, 0, i] = abs(t - x) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        rating[r] = int(float(row["rating"]))
    return {"mismatch": mism, "lag": lag, "ex_mask": mask, "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))

    w = pm.HalfNormal("w", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    # Conjunctive strictness: how much the worst-fitting example dominates.
    strictness = pm.HalfNormal("strictness", sigma=3.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    dist = pt.sum(w[None, :, None] * mismatch, axis=1)           # (n, K)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    beta = strictness + 0.01
    # Recency-weighted soft maximum of distances (log-mean-exp, stabilised).
    dmax = pt.max(dist * ex_mask, axis=1, keepdims=True)
    set_dist = pt.squeeze(dmax, 1) + pt.log(
        pt.sum(mem * pt.exp(beta * (dist - dmax)), axis=1) + 1e-12) / beta
    familiarity = pt.exp(-set_dist)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + gain * familiarity), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
