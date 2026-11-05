"""Normalized-attention featural similarity gradient.

People judge a target by how similar it is to the examples in a space of number
properties (parity, primality, squares, cubes, powers, multiples, final digit,
digits contained). The examples set a fixed budget of attention
that is shared out over the properties in proportion to a power of how
consistently the examples share each one; the target's distance is the
fraction of that attention budget falling on properties the target lacks, and
P(yes) falls off exponentially with that fraction (Shepard-style gradient),
scaled by the yes rate for a perfectly similar target. Because attention is a fixed budget, a lone
example spreads it over all of its many properties, so a target that shares most
of them (10 -> 80) is still judged highly similar, while several examples that
agree on one distinctive property (ends in 2) concentrate attention on it.
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
    return M


_M = _feature_matrix()
_LACK_T = (1.0 - _M)  # F x 100: target lacks feature


def prepare_observed(rows):
    sets, set_index = [], {}
    sidx, tidx, y = [], [], []
    for r in rows:
        s = str(r["number_set"])
        if s not in set_index:
            set_index[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        ex = sets[set_index[s]]
        t = int(float(r["target"]))
        sidx.append(set_index[s])
        tidx.append(t - 1)
        y.append(int(float(r["rating"])))
    share = np.array([_M[:, [e - 1 for e in ex]].mean(axis=1) for ex in sets])  # S x F
    return {
        "log_share": np.log(np.clip(share, 1e-6, 1.0)),
        "set_idx": np.array(sidx, dtype="int64"),
        "target_idx": np.array(tidx, dtype="int64"),
        "rating": np.array(y, dtype="int64"),
    }


with pm.Model() as model:
    log_share = pm.Data("log_share", np.zeros((1, _M.shape[0]), dtype="float64"))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    # how sharply attention concentrates on properties the examples share
    gamma = pm.LogNormal("gamma", mu=1.2, sigma=0.5)
    # steepness of the exponential generalization gradient
    c_feat = pm.HalfNormal("c_feat", sigma=5.0)
    # yes rate for a target that lacks none of the attended properties
    hi = pm.Beta("hi", alpha=10.0, beta=2.0)

    raw = pt.exp(gamma * log_share)                         # S x F
    attn = raw / pt.sum(raw, axis=1, keepdims=True)         # fixed attention budget
    d_tab = pt.dot(attn, pt.as_tensor_variable(_LACK_T))    # S x 100, in [0, 1]
    d_feat = d_tab[set_idx, target_idx]
    p_yes = pm.Deterministic("p_yes", pt.clip(hi * pt.exp(-c_feat * d_feat), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
