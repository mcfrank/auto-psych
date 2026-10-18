"""Exemplar similarity (GCM) with example-driven attention to shared features.

A refinement of exemplar_feature_similarity_gcm: people judge the target by its
summed exemplar similarity to the examples on a magnitude dimension and an
arithmetic-feature dimension (parity, divisibility by 3/4/5/10, prime, square,
power of 2, final digit, tens digit). The one change: attention across the
arithmetic features is set by the examples -- a feature on which all examples
agree gets extra attention (by an inferred amount) relative to features on
which they disagree, so breaking a property every example shares makes the
target much less similar to all of them.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_PRIMES = {n for n in range(2, 101) if all(n % k for k in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}
_POW2 = {2 ** k for k in range(0, 7)}
_NF = 10


def _feats(n):
    return [
        n % 2 == 0, n % 3 == 0, n % 4 == 0, n % 5 == 0, n % 10 == 0,
        n in _PRIMES, n in _SQUARES, n in _POW2, n % 10, n // 10,
    ]


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    ft = _feats(t)
    fx = [_feats(x) for x in xs]
    out = {}
    for f in range(_NF):
        out[f"shared{f}"] = 1.0 if all(v[f] == fx[0][f] for v in fx) else 0.0
    for i in range(4):
        have = i < len(xs)
        out[f"dmag{i}"] = abs(t - xs[i]) / 100.0 if have else 0.0
        out[f"mask{i}"] = 1.0 if have else 0.0
        for f in range(_NF):
            out[f"mis{i}_{f}"] = float(ft[f] != fx[i][f]) if have else 0.0
    return out


with pm.Model() as model:
    dmag = pt.stack([pm.Data(f"dmag{i}", np.zeros(1, dtype="float64")) for i in range(4)], axis=1)
    mask = pt.stack([pm.Data(f"mask{i}", np.zeros(1, dtype="float64")) for i in range(4)], axis=1)
    shared = pt.stack([pm.Data(f"shared{f}", np.zeros(1, dtype="float64")) for f in range(_NF)], axis=1)
    mis = pt.stack([
        pt.stack([pm.Data(f"mis{i}_{f}", np.zeros(1, dtype="float64")) for f in range(_NF)], axis=1)
        for i in range(4)
    ], axis=1)  # (trial, example, feature)

    c = pm.HalfNormal("c", sigma=10.0)
    w_mag = pm.Beta("w_mag", alpha=2.0, beta=2.0)
    # How strongly a feature shared by all examples draws attention.
    shift = pm.HalfNormal("shift", sigma=2.0)
    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=3.0)

    att = pt.exp(shift * shared)
    att = att / pt.sum(att, axis=1, keepdims=True)  # (trial, feature), sums to 1
    dfeat = pt.sum(mis * att[:, None, :], axis=2)
    dist = w_mag * dmag + (1.0 - w_mag) * dfeat
    sim = pt.sum(mask * pt.exp(-c * dist), axis=1)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(bias + gain * sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
