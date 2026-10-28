"""Featural exemplar similarity.

People do not induce a rule; they judge the target by its summed similarity to
the remembered examples (exemplar generalization). Similarity decays
exponentially with psychological distance, which combines magnitude distance and
the number of salient surface features (parity, final digit, decade,
divisibility by 3 and 5, primality, squareness, power of two) the two numbers
fail to share; higher average similarity makes "yes" more likely.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXN = 4
_PRIMES = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}
_POW2 = {2 ** k for k in range(0, 7)}


def _feats(n):
    return (n % 2, n % 10, min(n // 10, 9), n % 3 == 0, n % 5 == 0,
            n in _PRIMES, n in _SQUARES, n in _POW2)


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")][:MAXN]
    t = int(target)
    ft = _feats(t)
    out = {}
    for i in range(MAXN):
        if i < len(xs):
            fx = _feats(xs[i])
            out[f"mis_{i}"] = float(sum(a != b for a, b in zip(ft, fx)))
            out[f"mag_{i}"] = abs(t - xs[i]) / 100.0
            out[f"msk_{i}"] = 1.0
        else:
            out[f"mis_{i}"] = 0.0
            out[f"mag_{i}"] = 0.0
            out[f"msk_{i}"] = 0.0
    return out


with pm.Model() as model:
    mis = pt.stack([pm.Data(f"mis_{i}", np.zeros(1, dtype="float64")) for i in range(MAXN)], axis=1)
    mag = pt.stack([pm.Data(f"mag_{i}", np.zeros(1, dtype="float64")) for i in range(MAXN)], axis=1)
    msk = pt.stack([pm.Data(f"msk_{i}", np.zeros(1, dtype="float64")) for i in range(MAXN)], axis=1)

    # Attention to feature mismatches and to magnitude distance.
    w_feat = pm.HalfNormal("w_feat", sigma=1.0)
    w_mag = pm.HalfNormal("w_mag", sigma=10.0)
    # Mapping from mean similarity to a yes response.
    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    sim = pt.exp(-w_feat * mis - w_mag * mag) * msk
    mean_sim = pt.sum(sim, axis=1) / pt.maximum(pt.sum(msk, axis=1), 1.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(bias + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
