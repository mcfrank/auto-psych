"""Exemplar feature-similarity generalization.

People do not infer a rule; they judge a target by its average similarity to the
remembered examples (GCM-style exemplar generalization). Similarity between two
numbers decays exponentially with the share of simple arithmetic properties they
disagree on (parity, divisibility by 3-10, last digit, tens digit, prime, square,
power of two) plus their magnitude distance. No size principle, no joint
consistency requirement across examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4
_PRIMES = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}
_POW2 = {2 ** k for k in range(0, 7)}


def _props(n):
    return (
        [n % 2 == 0]
        + [n % m == 0 for m in range(3, 11)]
        + [n in _PRIMES, n in _SQUARES, n in _POW2]
        + [n % 10, (n // 10) % 10]
    )


def _feat_dist(a, b):
    pa, pb = _props(a), _props(b)
    return sum(1.0 for x, y in zip(pa, pb) if x != y) / len(pa)


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    out = {}
    for i in range(MAX_EX):
        if i < len(ex):
            out[f"fdist_{i}"] = _feat_dist(t, ex[i])
            out[f"mdist_{i}"] = abs(t - ex[i]) / 100.0
            out[f"present_{i}"] = 1.0
        else:
            out[f"fdist_{i}"] = 0.0
            out[f"mdist_{i}"] = 0.0
            out[f"present_{i}"] = 0.0
    return out


with pm.Model() as model:
    fd = pt.stack([pm.Data(f"fdist_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)], axis=1)
    md = pt.stack([pm.Data(f"mdist_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)], axis=1)
    pr = pt.stack([pm.Data(f"present_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)], axis=1)

    # sensitivity of similarity to feature mismatch, and relative weight of magnitude
    sens = pm.HalfNormal("sens", sigma=10.0)
    w_mag = pm.HalfNormal("w_mag", sigma=1.0)
    # mapping summed similarity to the yes probability
    alpha = pm.Normal("alpha", mu=-2.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=5.0)

    sim = pt.exp(-sens * (fd + w_mag * md)) * pr
    mean_sim = pt.sum(sim, axis=1) / pt.maximum(pt.sum(pr, axis=1), 1.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(alpha + beta * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
