"""Tversky ratio-model similarity to the examples' feature prototype.

People judge the target by its similarity to the prototype of the examples:
the familiar number properties (parity, multiples, squares/cubes/powers,
primes, final digit, decade) shared by all examples, plus their place on the
number line. Similarity is Tversky's ratio model: salience-weighted common
properties over common + alpha * prototype properties the target lacks +
beta * target properties the prototype lacks; numerical closeness to the
nearest example decays exponentially and counts as one more weighted
property. "Yes" grows with this similarity.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
_nums = np.arange(1, N + 1)


def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


# feature kinds: 0 parity, 1 multiples, 2 squares/cubes/powers, 3 primes,
# 4 final digit, 5 decade
_FEATS = []
_FEATS += [(0, _nums % 2 == 0), (0, _nums % 2 == 1)]
_FEATS += [(1, _nums % m == 0) for m in range(3, 13)]
_FEATS.append((2, np.isin(_nums, [k * k for k in range(1, 11)])))
_FEATS.append((2, np.isin(_nums, [k ** 3 for k in range(1, 5)])))
for b in range(2, 11):
    _FEATS.append((2, np.isin(_nums, [b ** k for k in range(1, 8) if b ** k <= N])))
_FEATS.append((3, np.array([_is_prime(int(n)) for n in _nums])))
_FEATS += [(4, _nums % 10 == d) for d in range(10)]
_FEATS += [(5, (_nums // 10) == t) for t in range(11)]
_KIND = np.array([k for k, _ in _FEATS])
_MEM = np.stack([m for _, m in _FEATS])  # (F, 100)
K = 6


def prepare_observed(rows):
    n = len(rows)
    common = np.zeros((n, K))
    dproto = np.zeros((n, K))
    dtarg = np.zeros((n, K))
    dist = np.zeros(n)
    y = np.zeros(n, dtype="int64")
    cache = {}
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        key = str(r["number_set"])
        if key not in cache:
            cache[key] = _MEM[:, [e - 1 for e in ex]].all(axis=1)
        proto = cache[key]
        has = _MEM[:, t - 1]
        for k in range(K):
            sel = _KIND == k
            common[i, k] = np.sum(proto[sel] & has[sel])
            dproto[i, k] = np.sum(proto[sel] & ~has[sel])
            dtarg[i, k] = np.sum(~proto[sel] & has[sel])
        dist[i] = min(abs(t - e) for e in ex) / 100.0
        y[i] = int(float(r["rating"]))
    return {"common_feats": common, "proto_only_feats": dproto,
            "target_only_feats": dtarg, "near_dist": dist, "rating": y}


with pm.Model() as model:
    common = pm.Data("common_feats", np.zeros((1, K)))
    dproto = pm.Data("proto_only_feats", np.zeros((1, K)))
    dtarg = pm.Data("target_only_feats", np.zeros((1, K)))
    near_dist = pm.Data("near_dist", np.zeros(1))

    # Salience of each property kind plus the magnitude dimension (last).
    salience = pm.Dirichlet("salience", a=np.full(K + 1, 2.0))
    alpha = pm.HalfNormal("alpha", sigma=1.0)   # weight of prototype-only props
    beta = pm.HalfNormal("beta", sigma=1.0)     # weight of target-only props
    mag_scale = pm.HalfNormal("mag_scale", sigma=0.2)  # closeness decay (/100)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    w = salience[:K]
    w_mag = salience[K]
    close = pt.exp(-near_dist / (mag_scale + 0.01))
    c = pt.dot(common, w) + w_mag * close
    a = pt.dot(dproto, w) + w_mag * (1.0 - close)
    b = pt.dot(dtarg, w)
    sim = c / (c + alpha * a + beta * b + 1e-6)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(intercept + gain * sim), 1e-6, 1 - 1e-6))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
