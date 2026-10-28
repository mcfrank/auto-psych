"""Exemplar similarity (GCM) hypothesis.

People do not induce a rule; they judge the target by its similarity to the
stored examples. Each number is represented by its magnitude and its salient
arithmetic/digit properties (parity, primality, square, cube, multiples, last
and tens digit). Similarity decays exponentially with an attention-weighted
distance in this space, and the probability of "yes" grows with the target's
mean similarity to the examples (Luce choice ratio against a criterion).
"""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4
_SQUARES = {i * i for i in range(1, 11)}
_CUBES = {i ** 3 for i in range(1, 5)}


def _is_prime(n):
    if n < 2:
        return False
    return all(n % d for d in range(2, int(math.isqrt(n)) + 1))


def _powers(b):
    s, v = set(), b
    while v <= 100:
        s.add(v)
        v *= b
    return s


_POW = {b: _powers(b) for b in (2, 3, 5)}


def _props(n):
    f = [n % 2 == 0, _is_prime(n), n in _SQUARES, n in _CUBES]
    f += [n % m == 0 for m in range(3, 11)]
    f += [n in _POW[b] for b in (2, 3, 5)]
    f += [n % 10 == d for d in range(10)]
    f += [min(n // 10, 9) == d for d in range(10)]
    return np.array(f, dtype=float)


_NF = len(_props(1))


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")][:MAX_EX]
    t = int(target)
    pt_ = _props(t)
    out = {}
    for i in range(MAX_EX):
        if i < len(ex):
            out[f"mag_d{i}"] = abs(t - ex[i]) / 100.0
            out[f"feat_d{i}"] = float(np.abs(pt_ - _props(ex[i])).sum()) / _NF
            out[f"mask{i}"] = 1.0
        else:
            out[f"mag_d{i}"] = 0.0
            out[f"feat_d{i}"] = 0.0
            out[f"mask{i}"] = 0.0
    return out


with pm.Model() as model:
    mag = pt.stack([pm.Data(f"mag_d{i}", np.zeros(1)) for i in range(MAX_EX)], axis=0)
    feat = pt.stack([pm.Data(f"feat_d{i}", np.zeros(1)) for i in range(MAX_EX)], axis=0)
    mask = pt.stack([pm.Data(f"mask{i}", np.ones(1)) for i in range(MAX_EX)], axis=0)

    # Sensitivity of similarity to distance, attention to magnitude vs
    # properties, and the response criterion of the Luce choice rule.
    c = pm.HalfNormal("c", sigma=10.0)
    w = pm.Beta("w_mag", alpha=2.0, beta=2.0)
    log_k = pm.Normal("log_k", mu=-1.0, sigma=1.5)

    dist = w * mag + (1.0 - w) * feat
    sim = pt.exp(-c * dist) * mask
    S = sim.sum(axis=0) / mask.sum(axis=0)
    k = pt.exp(log_k)
    p_yes = pm.Deterministic("p_yes", pt.clip(S / (S + k), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
