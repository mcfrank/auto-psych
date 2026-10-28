"""Common-factor-aware recency-weighted exemplar similarity.

Refinement of recency_weighted_exemplar_similarity: people judge a target by
its recency-weighted exponential similarity to the remembered examples
(mismatches on magnitude, final digit, parity and pairwise divisibility), but
when a multi-example set shares a greatest common divisor g >= 3 they also
encode that shared factor as a property of the whole set, so a target that is
not a multiple of g is farther from every example. The single change is this
set-level common-factor mismatch dimension in the distance.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def prepare_observed(rows):
    n = len(rows)
    mag = np.zeros((n, MAXK))
    digit = np.ones((n, MAXK))
    parity = np.ones((n, MAXK))
    nodiv = np.ones((n, MAXK))
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    gcd_mis = np.zeros(n)
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")]
        t = int(row["target"])
        k = len(ex)
        for i, x in enumerate(ex[:MAXK]):
            mag[r, i] = abs(t - x) / 100.0
            digit[r, i] = 0.0 if t % 10 == x % 10 else 1.0
            parity[r, i] = 0.0 if t % 2 == x % 2 else 1.0
            nodiv[r, i] = 0.0 if (t % x == 0 or x % t == 0) else 1.0
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        if k >= 2:
            g = reduce(math.gcd, ex)
            if g >= 3 and t % g != 0:
                gcd_mis[r] = 1.0
        rating[r] = int(float(row["rating"]))
    return {"mag_dist": mag, "digit_mismatch": digit, "parity_mismatch": parity,
            "div_mismatch": nodiv, "lag": lag, "ex_mask": mask,
            "gcd_mismatch": gcd_mis, "rating": rating}


with pm.Model() as model:
    mag_dist = pm.Data("mag_dist", np.zeros((1, MAXK)))
    digit_mismatch = pm.Data("digit_mismatch", np.zeros((1, MAXK)))
    parity_mismatch = pm.Data("parity_mismatch", np.zeros((1, MAXK)))
    div_mismatch = pm.Data("div_mismatch", np.zeros((1, MAXK)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    gcd_mismatch = pm.Data("gcd_mismatch", np.zeros(1))

    # Attention weights on the dimensions of psychological distance.
    w_mag = pm.HalfNormal("w_mag", sigma=10.0)
    w_digit = pm.HalfNormal("w_digit", sigma=2.0)
    w_parity = pm.HalfNormal("w_parity", sigma=2.0)
    w_div = pm.HalfNormal("w_div", sigma=2.0)
    # Attention to the set-level shared common factor (the one change).
    w_gcd = pm.HalfNormal("w_gcd", sigma=2.0)
    # Memory retention per step back in the presentation order.
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    # Mapping from summed similarity to the yes probability.
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    dist = (w_mag * mag_dist + w_digit * digit_mismatch
            + w_parity * parity_mismatch + w_div * div_mismatch
            + w_gcd * gcd_mismatch[:, None])
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + gain * familiarity), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
