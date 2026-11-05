"""Featural-exemplar similarity (GCM-style) generalization.

People do not pick a single best-fitting rule; they judge a target by its graded
exemplar similarity to the examples, where two numbers are similar to the extent
that they are close on the number line and share mathematical/digit features
(parity, last digit, multiples, powers, primality, contained digits). "Yes"
rises with the target's average similarity to the examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def _features(n):
    f = set()
    f.add("odd" if n % 2 else "even")
    if n > 1 and all(n % k for k in range(2, int(n ** 0.5) + 1)):
        f.add("prime")
    if round(n ** 0.5) ** 2 == n:
        f.add("square")
    if round(n ** (1 / 3)) ** 3 == n:
        f.add("cube")
    for b in range(2, 11):
        p = 1
        while p < n:
            p *= b
        if p == n:
            f.add(f"pow{b}")
    for m in range(3, 13):
        if n % m == 0:
            f.add(f"mult{m}")
    f.add(f"end{n % 10}")
    for d in set(str(n)):
        f.add(f"has{d}")
    return f


_FEATS = {n: _features(n) for n in range(1, 101)}


def compute_features(number_set, target):
    xs = [int(x) for x in str(number_set).split(",")][:MAXK]
    t = int(target)
    ft = _FEATS[t]
    out = {}
    for i in range(MAXK):
        if i < len(xs):
            fx = _FEATS[xs[i]]
            fd = 1.0 - len(ft & fx) / len(ft | fx)
            nd = abs(t - xs[i]) / 100.0
            m = 1.0
        else:
            fd, nd, m = 1.0, 1.0, 0.0
        out[f"fdist{i}"] = fd
        out[f"ndist{i}"] = nd
        out[f"mask{i}"] = m
    return out


with pm.Model() as model:
    fd = pt.stack([pm.Data(f"fdist{i}", np.zeros(1, dtype="float64")) for i in range(MAXK)], axis=1)
    nd = pt.stack([pm.Data(f"ndist{i}", np.zeros(1, dtype="float64")) for i in range(MAXK)], axis=1)
    mk = pt.stack([pm.Data(f"mask{i}", np.zeros(1, dtype="float64")) for i in range(MAXK)], axis=1)

    # attention weight on featural vs numeric dimension
    w_feat = pm.Beta("w_feat", alpha=2.0, beta=2.0)
    # similarity gradient specificity
    c = pm.HalfNormal("c", sigma=10.0)
    # response mapping
    alpha = pm.Normal("alpha", mu=-2.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=5.0)

    dist = w_feat * fd + (1.0 - w_feat) * nd
    sim = pt.exp(-c * dist)
    mean_sim = pt.sum(sim * mk, axis=1) / pt.sum(mk, axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(alpha + beta * mean_sim), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
