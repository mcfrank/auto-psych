"""Exemplar (GCM) similarity hypothesis.

People do not entertain explicit rules; they judge the target by its average
psychological similarity to the remembered examples. Similarity decays
exponentially with distance in a feature space describing each number by its
magnitude and by its surface properties (parity, last digit, tens digit,
divisibility by 3/4/5, prime, square), with attention split between magnitude
and properties. Greater mean similarity -> more likely "yes".
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_PRIMES = {n for n in range(2, 101) if all(n % k for k in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}


def _props(n):
    return (n % 2, n % 10, (n // 10) % 10, n % 3 == 0, n % 4 == 0, n % 5 == 0,
            n in _PRIMES, n in _SQUARES)


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    pt_ = _props(t)
    out = {}
    for i in range(4):
        if i < len(ex):
            x = ex[i]
            px = _props(x)
            out[f"magd_{i}"] = abs(t - x) / 100.0
            out[f"propd_{i}"] = sum(a != b for a, b in zip(pt_, px)) / len(px)
            out[f"mask_{i}"] = 1.0
        else:
            out[f"magd_{i}"] = 0.0
            out[f"propd_{i}"] = 0.0
            out[f"mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    magd = pt.stack([pm.Data(f"magd_{i}", np.zeros(1)) for i in range(4)], axis=1)
    propd = pt.stack([pm.Data(f"propd_{i}", np.zeros(1)) for i in range(4)], axis=1)
    mask = pt.stack([pm.Data(f"mask_{i}", np.zeros(1)) for i in range(4)], axis=1)

    # attention to magnitude (vs. surface properties)
    w_mag = pm.Beta("w_mag", alpha=2.0, beta=2.0)
    # similarity sensitivity
    c = pm.LogNormal("c", mu=1.5, sigma=0.7)
    # response mapping: gain and threshold on mean similarity
    gain = pm.LogNormal("gain", mu=1.5, sigma=0.7)
    theta = pm.Normal("theta", mu=0.3, sigma=0.3)

    dist = w_mag * magd + (1.0 - w_mag) * propd
    sim = pt.exp(-c * dist) * mask
    mean_sim = pt.sum(sim, axis=1) / pt.maximum(pt.sum(mask, axis=1), 1.0)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(gain * (mean_sim - theta) * 5.0), 1e-6, 1 - 1e-6)
    )
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
