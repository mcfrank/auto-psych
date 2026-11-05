"""Coherence-driven digit attention in exemplar similarity.

People judge a target by its summed similarity to the remembered examples
(magnitude and final digit), but how much attention they give to the final
digit is set by reading the examples: the more the examples agree with one
another on their final digit (fraction of example pairs sharing it), the more
attention shifts onto it. Sets that agree on a final digit produce narrow,
digit-bound generalization; sets with mixed final digits produce broad
generalization by magnitude.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")][:MAXK]
    t = int(target)
    pairs = list(itertools.combinations(ex, 2))
    agree = (sum(a % 10 == b % 10 for a, b in pairs) / len(pairs)) if pairs else 0.0
    out = {"digit_agreement": float(agree)}
    for k in range(MAXK):
        if k < len(ex):
            out[f"dist_{k}"] = abs(t - ex[k]) / 10.0
            out[f"unit_diff_{k}"] = 0.0 if t % 10 == ex[k] % 10 else 1.0
            out[f"present_{k}"] = 1.0
        else:
            out[f"dist_{k}"] = 0.0
            out[f"unit_diff_{k}"] = 0.0
            out[f"present_{k}"] = 0.0
    return out


with pm.Model() as model:
    dist = pt.stack([pm.Data(f"dist_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    udiff = pt.stack([pm.Data(f"unit_diff_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    present = pt.stack([pm.Data(f"present_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    agree = pm.Data("digit_agreement", np.zeros(1, dtype="float64"))

    c_mag = pm.HalfNormal("c_mag", sigma=2.0)          # magnitude gradient (per 10 units)
    c_unit = pm.HalfNormal("c_unit", sigma=3.0)        # baseline digit-mismatch penalty
    c_shift = pm.HalfNormal("c_shift", sigma=3.0)      # extra digit attention from example agreement
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)

    digit_w = (c_unit + c_shift * agree)[:, None]
    log_sim = -c_mag * dist - digit_w * udiff
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(alpha + beta * log_summed))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
