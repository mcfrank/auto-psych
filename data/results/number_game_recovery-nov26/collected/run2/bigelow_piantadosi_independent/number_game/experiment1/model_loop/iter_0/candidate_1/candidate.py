"""Digit/parity exemplar similarity (GCM-style).

People do not induce a hidden rule; they judge a target by its exemplar
similarity to the examples shown, where two numbers are similar to the extent
they share surface features (same parity, same final digit) and are close in
magnitude, and "yes" grows with the target's average similarity to the
examples. Disagreement with the rule-based Bayesian model lives on targets that
partially match the examples' final digit or parity without fitting any whole
rule, and on magnitude-near targets that break the parity/digit pattern.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")][:K]
    t = int(target)
    out = {}
    for i in range(K):
        if i < len(xs):
            x = xs[i]
            out[f"par_mis_{i}"] = float((x % 2) != (t % 2))
            out[f"dig_mis_{i}"] = float((x % 10) != (t % 10))
            out[f"mag_{i}"] = abs(x - t) / 10.0
            out[f"mask_{i}"] = 1.0
        else:
            out[f"par_mis_{i}"] = 0.0
            out[f"dig_mis_{i}"] = 0.0
            out[f"mag_{i}"] = 0.0
            out[f"mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    par = pt.stack([pm.Data(f"par_mis_{i}", np.zeros(1)) for i in range(K)], axis=1)
    dig = pt.stack([pm.Data(f"dig_mis_{i}", np.zeros(1)) for i in range(K)], axis=1)
    mag = pt.stack([pm.Data(f"mag_{i}", np.zeros(1)) for i in range(K)], axis=1)
    mask = pt.stack([pm.Data(f"mask_{i}", np.zeros(1)) for i in range(K)], axis=1)

    # Attention weights on each similarity dimension.
    w_par = pm.HalfNormal("w_par", sigma=2.0)
    w_dig = pm.HalfNormal("w_dig", sigma=2.0)
    w_mag = pm.HalfNormal("w_mag", sigma=2.0)
    # Response mapping from mean similarity to yes.
    bias = pm.Normal("bias", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    dist = w_par * par + w_dig * dig + w_mag * mag
    sim = pt.exp(-dist) * mask
    mean_sim = pt.sum(sim, axis=1) / pt.maximum(pt.sum(mask, axis=1), 1.0)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + gain * mean_sim), 1e-6, 1 - 1e-6)
    )
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
