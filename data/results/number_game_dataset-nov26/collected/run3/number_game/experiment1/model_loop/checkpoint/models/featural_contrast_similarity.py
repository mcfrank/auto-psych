"""Featural contrast-model similarity (Tversky).

People do not pick a single rule; they judge a target by featural similarity to
the examples. Each number is a bundle of properties (parity, primality, squares,
cubes, powers, multiples, final digit, contained digits, decade). The yes
probability rises with the (distinctiveness-weighted) share of the examples'
properties the target also has, and falls with the target's properties the
examples lack.
"""
import math
import numpy as np
import pymc as pm


def _features():
    N = range(1, 101)
    feats = []
    feats.append({n for n in N if n % 2 == 1})
    feats.append({n for n in N if n % 2 == 0})
    feats.append({n for n in N if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))})
    feats.append({k * k for k in range(1, 11)})
    feats.append({k ** 3 for k in range(1, 5)})
    for b in range(2, 11):
        feats.append({b ** k for k in range(1, 8) if b ** k <= 100})
    for m in list(range(3, 10)) + [11, 12]:
        feats.append({n for n in N if n % m == 0})
    for d in range(10):
        feats.append({n for n in N if n % 10 == d})
    for d in range(10):
        feats.append({n for n in N if str(d) in str(n)})
    for lo in range(1, 101, 10):
        feats.append(set(range(lo, lo + 10)))
    feats = [f for f in feats if 0 < len(f) < 100]
    M = np.array([[1.0 if n in f else 0.0 for n in N] for f in feats])
    w = np.array([math.log(100.0 / len(f)) for f in feats])
    return M, w


_M, _W = _features()


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    share = _M[:, [e - 1 for e in ex]].mean(axis=1)  # fraction of examples having f
    tf = _M[:, t - 1]
    denom_c = float(np.sum(_W * share)) + 1e-9
    common = float(np.sum(_W * share * tf)) / denom_c
    denom_t = float(np.sum(_W * tf)) + 1e-9
    distinct = float(np.sum(_W * (1.0 - share) * tf)) / denom_t
    return {"feat_common": common, "feat_distinct": distinct}


with pm.Model() as model:
    common = pm.Data("feat_common", np.zeros(1, dtype="float64"))
    distinct = pm.Data("feat_distinct", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-1.0, sigma=2.0)
    theta = pm.HalfNormal("theta", sigma=5.0)   # weight of common features
    alpha = pm.HalfNormal("alpha", sigma=5.0)   # penalty for target-only features

    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + theta * common - alpha * distinct)
    )
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
