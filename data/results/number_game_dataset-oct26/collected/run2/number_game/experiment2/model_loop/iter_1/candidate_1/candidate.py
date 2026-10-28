"""Drifting response criterion on simple exemplar familiarity.

People judge a target by its plain recency-weighted similarity to the
remembered examples (closeness in size, shared final digit, shared parity,
graded shared factor), but the criterion they compare that familiarity against
relaxes steadily over the session, so the same familiarity earns "yes" more
often late than early. The single claim is a decision-level criterion that
drifts linearly with trial position; the inference is fixed-attention
exemplar similarity.
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
    session_pos = np.zeros(n)
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
        # Session position, centred and scaled to roughly [-0.5, 0.5].
        session_pos[r] = float(row["trial_index"]) / 450.0 - 0.5
        rating[r] = int(float(row["rating"]))
    return {"mismatch": mism, "lag": lag, "ex_mask": mask,
            "session_pos": session_pos, "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    session_pos = pm.Data("session_pos", np.zeros(1))

    # Fixed attention on each dimension (magnitude on a 0-1 scale).
    w = pm.HalfNormal("w", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # Criterion at mid-session, and how far it relaxes from start to end.
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    drift = pm.Normal("drift", mu=0.0, sigma=1.0)

    dist = pt.sum(w[None, :, None] * mismatch, axis=1)           # (n, K)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    logit = bias + drift * session_pos + gain * familiarity
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
