"""Session-broadening similarity gradient.

People judge a target by its recency-weighted exponential similarity to the
remembered examples, where psychological distance is a weighted sum of
mismatches in size, final digit, parity and graded shared factor, but their
generalization gradient broadens as the session goes on: every mismatch is
scaled down by a factor that shrinks with trial position, so the same distance
yields more similarity late in the session than early. This produces a drift
toward "yes" concentrated on targets dissimilar to the examples.
"""
import math

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
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    progress = np.zeros(n)
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
        progress[r] = min(max(float(row["trial_index"]), 0.0), SESSION_LEN) / SESSION_LEN
        rating[r] = int(float(row["rating"]))
    return {"mismatch": mism, "lag": lag, "ex_mask": mask,
            "session_progress": progress, "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    session_progress = pm.Data("session_progress", np.zeros(1))

    # Attention on each mismatch dimension at the start of the session.
    w = pm.HalfNormal("w", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # How much the generalization gradient broadens from session start to end.
    broadening = pm.HalfNormal("broadening", sigma=1.0)

    scale = pt.exp(-broadening * session_progress)                  # (n,)
    dist = scale[:, None] * pt.sum(w[None, :, None] * mismatch, axis=1)  # (n, K)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + gain * familiarity), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
