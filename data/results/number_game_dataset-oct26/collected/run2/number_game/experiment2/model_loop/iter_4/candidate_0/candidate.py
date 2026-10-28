"""Spread-scaled size exemplar model.

People estimate how variable the program's output is from how spread out the
examples are, and generalize in size accordingly: a target is judged by its
recency-weighted exponential similarity to the remembered examples (size, final
digit, parity, graded shared factor), but the size gradient around every
example is stretched in proportion to the spread (range) of the examples.
Tight sets are generalized narrowly in size, widely spread sets broadly.
"""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
NDIM = 4  # size, final digit, parity, graded shared factor


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
    spread = np.zeros(n)
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")][:MAXK]
        t = int(row["target"])
        k = len(ex)
        spread[r] = (max(ex) - min(ex)) / 100.0
        for i, x in enumerate(ex):
            mism[r, 0, i] = abs(t - x) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        rv = row.get("rating")
        rating[r] = int(float(rv)) if rv is not None and str(rv).strip() != "" else 0
    return {"mismatch": mism, "lag": lag, "ex_mask": mask,
            "example_spread": spread, "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    example_spread = pm.Data("example_spread", np.zeros(1))

    w = pm.HalfNormal("w", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # How strongly the examples' spread widens the size gradient (the claim).
    spread_gain = pm.HalfNormal("spread_gain", sigma=5.0)

    width = 1.0 + spread_gain * example_spread
    size_dist = mismatch[:, 0, :] / width[:, None]
    dist = (w[0] * size_dist + w[1] * mismatch[:, 1, :]
            + w[2] * mismatch[:, 2, :] + w[3] * mismatch[:, 3, :])
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + gain * familiarity), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
