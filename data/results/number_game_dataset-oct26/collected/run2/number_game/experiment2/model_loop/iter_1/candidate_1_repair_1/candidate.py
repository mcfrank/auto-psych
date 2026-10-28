"""Lapse-to-guessing decision rule on simple exemplar familiarity.

People judge a target by its plain recency-weighted similarity to the
remembered examples (closeness in size, shared final digit, shared parity,
graded shared factor) and answer from that familiarity, but on some fraction
of trials they ignore the examples and simply guess "yes" at their own
guessing rate. The single claim is this decision-level lapse process: a
constant floor of "yes" to unlike targets and ceiling of "no" to near-identical
ones, the same for every set and target.
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
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")][:MAXK]
        t = int(float(row["target"]))
        k = len(ex)
        for i, x in enumerate(ex):
            mism[r, 0, i] = abs(t - x) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        rv = row.get("rating", 0)
        rating[r] = int(float(rv)) if rv not in (None, "") else 0
    return {"mismatch": mism, "lag": lag, "ex_mask": mask, "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))

    # Fixed attention on each dimension (magnitude on a 0-1 scale).
    w = pm.HalfNormal("w", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    # Lapse rate and the yes-rate of guesses on lapsed trials.
    lapse = pm.Beta("lapse", alpha=1.5, beta=8.0)
    guess = pm.Beta("guess", alpha=2.0, beta=2.0)

    dist = pt.sum(w[None, :, None] * mismatch, axis=1)           # (n, K)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    p_engaged = pm.math.sigmoid(bias + gain * familiarity)
    p = (1.0 - lapse) * p_engaged + lapse * guess
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
