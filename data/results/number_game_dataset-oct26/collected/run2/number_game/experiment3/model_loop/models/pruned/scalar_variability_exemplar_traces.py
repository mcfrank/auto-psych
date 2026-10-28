"""Scalar-variability exemplar traces.

People judge a target by its recency-weighted similarity to the stored
examples (closeness in size, shared final digit, shared parity, shared
factors), but each example is stored as an approximate magnitude whose
fuzziness grows in proportion to its size (scalar variability of the
approximate number system). The size gradient is therefore sharp around small
examples and broad around large ones, so equal linear gaps beyond small
examples are rejected more than beyond large ones.
"""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def _factor_mismatch(t, x):
    m = min(t, x)
    if m <= 1:
        return 0.0
    return 1.0 - math.log(math.gcd(t, x)) / math.log(m)


def prepare_observed(rows):
    n = len(rows)
    size_gap = np.zeros((n, MAXK))
    ex_size = np.ones((n, MAXK))
    other = np.ones((n, 3, MAXK))
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")][:MAXK]
        t = int(row["target"])
        k = len(ex)
        for i, x in enumerate(ex):
            size_gap[r, i] = abs(t - x) / 100.0
            ex_size[r, i] = x / 100.0
            other[r, 0, i] = 0.0 if t % 10 == x % 10 else 1.0
            other[r, 1, i] = 0.0 if t % 2 == x % 2 else 1.0
            other[r, 2, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        rv = row.get("rating")
        rating[r] = int(float(rv)) if rv is not None and str(rv).strip() != "" else 0
    return {"size_gap": size_gap, "ex_size": ex_size, "other_mismatch": other,
            "lag": lag, "ex_mask": mask, "rating": rating}


with pm.Model() as model:
    size_gap = pm.Data("size_gap", np.zeros((1, MAXK)))
    ex_size = pm.Data("ex_size", np.ones((1, MAXK)))
    other_mismatch = pm.Data("other_mismatch", np.ones((1, 3, MAXK)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))

    # Attention to size (per 100 units at an example of size 100) and to the
    # final digit, parity and shared factor.
    w_size = pm.HalfNormal("w_size", sigma=10.0)
    w_other = pm.HalfNormal("w_other", sigma=2.0, shape=3)
    # Floor of trace fuzziness relative to the scalar part: small -> pure
    # Weber scaling, large -> fuzziness the same for every magnitude.
    log_floor = pm.Normal("log_floor", mu=-1.5, sigma=1.0)
    floor = pt.exp(log_floor)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    # Trace width proportional to the example's magnitude (plus a floor),
    # normalized to 1 at magnitude 100.
    width = (ex_size + floor) / (1.0 + floor)
    dist = (w_size * size_gap / width
            + pt.sum(w_other[None, :, None] * other_mismatch, axis=1))
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + gain * familiarity), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
