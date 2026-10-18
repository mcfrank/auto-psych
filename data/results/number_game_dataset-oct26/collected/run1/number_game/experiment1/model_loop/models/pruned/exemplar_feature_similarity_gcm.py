"""Exemplar similarity (GCM) hypothesis.

People do not induce a rule; they judge the target by its exemplar similarity
to the individual examples shown. Each number is represented on a magnitude
dimension and an arithmetic-feature dimension (parity, divisibility by
3/4/5/10, final digit, tens digit, prime, square, power of 2). Similarity to
each example decays exponentially with attention-weighted distance; the
probability of "yes" grows with the summed similarity to the examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_PRIMES = {n for n in range(2, 101) if all(n % k for k in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}
_POW2 = {2 ** k for k in range(0, 7)}


def _feats(n):
    return [
        n % 2 == 0, n % 3 == 0, n % 4 == 0, n % 5 == 0, n % 10 == 0,
        n in _PRIMES, n in _SQUARES, n in _POW2,
    ]


def _mismatch(a, b):
    fa, fb = _feats(a), _feats(b)
    m = sum(x != y for x, y in zip(fa, fb))
    m += (a % 10) != (b % 10)
    m += (a // 10) != (b // 10)
    return m / 10.0


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    out = {}
    for i in range(4):
        if i < len(xs):
            out[f"dmag{i}"] = abs(t - xs[i]) / 100.0
            out[f"dfeat{i}"] = _mismatch(t, xs[i])
            out[f"mask{i}"] = 1.0
        else:
            out[f"dmag{i}"] = 0.0
            out[f"dfeat{i}"] = 0.0
            out[f"mask{i}"] = 0.0
    return out


with pm.Model() as model:
    dmag = pt.stack([pm.Data(f"dmag{i}", np.zeros(1, dtype="float64")) for i in range(4)], axis=1)
    dfeat = pt.stack([pm.Data(f"dfeat{i}", np.zeros(1, dtype="float64")) for i in range(4)], axis=1)
    mask = pt.stack([pm.Data(f"mask{i}", np.zeros(1, dtype="float64")) for i in range(4)], axis=1)

    # Overall sensitivity (specificity) and attention share on magnitude.
    c = pm.HalfNormal("c", sigma=10.0)
    w_mag = pm.Beta("w_mag", alpha=2.0, beta=2.0)
    # Response mapping from summed similarity to yes probability.
    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=3.0)

    dist = w_mag * dmag + (1.0 - w_mag) * dfeat
    sim = pt.sum(mask * pt.exp(-c * dist), axis=1)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(bias + gain * sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
