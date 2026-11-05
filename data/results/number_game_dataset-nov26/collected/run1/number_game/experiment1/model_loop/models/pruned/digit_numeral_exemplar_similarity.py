"""Digit-numeral exemplar similarity.

People do not induce a rule or interval; they judge the target by its summed
similarity to the examples treated as two-digit numerals. Similarity to an
example decays exponentially with the distance between tens digits and is
multiplied by a fixed factor when the final digits differ. Endorsement is a
logistic function of the mean similarity to the examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4


def _digits(n):
    return n // 10, n % 10


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")][:K]
    t = int(target)
    tt, to = _digits(t)
    out = {}
    for k in range(K):
        if k < len(ex):
            et, eo = _digits(ex[k])
            out[f"dtens_{k}"] = float(abs(tt - et))
            out[f"odiff_{k}"] = 0.0 if to == eo else 1.0
            out[f"mask_{k}"] = 1.0
        else:
            out[f"dtens_{k}"] = 0.0
            out[f"odiff_{k}"] = 0.0
            out[f"mask_{k}"] = 0.0
    return out


with pm.Model() as model:
    dtens = pt.stack([pm.Data(f"dtens_{k}", np.zeros(1, dtype="float64")) for k in range(K)], axis=1)
    odiff = pt.stack([pm.Data(f"odiff_{k}", np.zeros(1, dtype="float64")) for k in range(K)], axis=1)
    mask = pt.stack([pm.Data(f"mask_{k}", np.zeros(1, dtype="float64")) for k in range(K)], axis=1)

    # decay per decade of tens-digit distance
    w_tens = pm.HalfNormal("w_tens", sigma=1.0)
    # log-penalty for a differing final digit
    w_ones = pm.HalfNormal("w_ones", sigma=2.0)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    sim = pt.exp(-w_tens * dtens - w_ones * odiff) * mask
    mean_sim = pt.sum(sim, axis=1) / pt.maximum(pt.sum(mask, axis=1), 1.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
