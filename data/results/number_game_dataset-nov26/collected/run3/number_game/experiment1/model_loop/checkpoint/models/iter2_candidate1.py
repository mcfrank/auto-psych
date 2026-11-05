"""Attention-weighted exponential generalization gradient.

People judge a target by Shepard-style exponential generalization in a
psychological space whose dimensions are weighted by selective attention set by
the examples: attention goes to properties the examples agree on (more strongly
the more consistently they share a property and the more distinctive it is), and
to magnitude in proportion to how tightly the examples cluster. P(yes) falls off
exponentially with the target's attention-weighted distance from the examples.
"""
import math
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _feature_matrix():
    N = range(1, 101)
    feats = [
        {n for n in N if n % 2 == 1},
        {n for n in N if n % 2 == 0},
        {n for n in N if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))},
        {k * k for k in range(1, 11)},
        {k ** 3 for k in range(1, 5)},
    ]
    for b in range(2, 11):
        feats.append({b ** k for k in range(1, 8) if b ** k <= 100})
    for m in range(3, 13):
        feats.append({n for n in N if n % m == 0})
    for d in range(10):
        feats.append({n for n in N if n % 10 == d})
    for d in range(10):
        feats.append({n for n in N if str(d) in str(n)})
    feats = [f for f in feats if 0 < len(f) < 100]
    M = np.array([[1.0 if n in f else 0.0 for n in N] for f in feats])  # F x 100
    idf = np.array([math.log(100.0 / len(f)) for f in feats])
    return M, idf


_M, _IDF = _feature_matrix()


def prepare_observed(rows):
    sets = []
    set_index = {}
    sidx, tidx, dmag, y = [], [], [], []
    for r in rows:
        s = str(r["number_set"])
        if s not in set_index:
            set_index[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        ex = sets[set_index[s]]
        t = int(float(r["target"]))
        lo, hi = min(ex), max(ex)
        out = max(0, lo - t, t - hi)
        sidx.append(set_index[s])
        tidx.append(t - 1)
        dmag.append(out / (hi - lo + 10.0))
        y.append(int(float(r["rating"])))
    share = np.array([_M[:, [e - 1 for e in ex]].mean(axis=1) for ex in sets])  # S x F
    return {
        "log_share": np.log(np.clip(share, 1e-6, 1.0)),
        "set_idx": np.array(sidx, dtype="int64"),
        "target_idx": np.array(tidx, dtype="int64"),
        "mag_dist": np.array(dmag, dtype="float64"),
        "rating": np.array(y, dtype="int64"),
    }


_LACK = (1.0 - _M).T  # 100 x F: target lacks feature

with pm.Model() as model:
    log_share = pm.Data("log_share", np.zeros((1, _M.shape[0]), dtype="float64"))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))
    mag_dist = pm.Data("mag_dist", np.zeros(1, dtype="float64"))

    # how sharply attention concentrates on properties all examples share
    gamma = pm.Gamma("gamma", alpha=3.0, beta=0.5)
    # gradient steepness on the featural and magnitude dimensions
    c_feat = pm.HalfNormal("c_feat", sigma=2.0)
    c_mag = pm.HalfNormal("c_mag", sigma=2.0)
    # response range: guessing floor and ceiling
    floor = pm.Beta("floor", alpha=1.5, beta=10.0)
    ceil = pm.Beta("ceil", alpha=10.0, beta=1.5)

    attn = pt.exp(gamma * log_share) * _IDF[None, :]          # S x F
    d_feat_tab = pt.dot(attn, pt.as_tensor_variable(_LACK.T))  # S x 100
    d_feat = d_feat_tab[set_idx, target_idx]
    sim = pt.exp(-c_feat * d_feat - c_mag * mag_dist)
    lo = floor
    hi = floor + (1.0 - floor) * ceil
    p_yes = pm.Deterministic("p_yes", pt.clip(lo + (hi - lo) * sim, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
