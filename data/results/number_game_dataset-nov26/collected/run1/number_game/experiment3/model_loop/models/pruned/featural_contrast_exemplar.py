"""Featural contrast-model exemplar (Tversky's contrast model applied to numbers).

People represent each number as a bundle of familiar arithmetic and notational
properties (odd/even, multiple of 3..12, square, cube, power of two, prime,
final digit, decade) and judge a target by its featural similarity to each
example: shared properties raise similarity, and properties that only the
target has, or only the example has, lower it, each property weighted by how
distinctive (rare among 1-100) it is, with closeness in magnitude as one more
graded attribute. Support is pooled over the examples and turned into a yes
probability; there is no coincidence sharpening or rule reading at the set
level, only pairwise feature matching.
"""
from math import log

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def _is_prime(n):
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n ** 0.5) + 1))


def _features(n):
    f = set()
    f.add("even" if n % 2 == 0 else "odd")
    for k in range(3, 13):
        if n % k == 0:
            f.add(f"mult{k}")
    if round(n ** 0.5) ** 2 == n:
        f.add("square")
    if round(n ** (1 / 3)) ** 3 == n:
        f.add("cube")
    if n & (n - 1) == 0:
        f.add("pow2")
    if _is_prime(n):
        f.add("prime")
    f.add(f"end{n % 10}")
    f.add(f"dec{min(n // 10, 9)}")
    return f


_FEATS = {n: _features(n) for n in range(1, 101)}
_SIZE = {}
for _n in range(1, 101):
    for _f in _FEATS[_n]:
        _SIZE[_f] = _SIZE.get(_f, 0) + 1
# Salience of a property: its information content among 1-100.
_W = {f: log(100.0 / s) for f, s in _SIZE.items()}


def _contrast(t, e):
    ft, fe = _FEATS[t], _FEATS[e]
    common = sum(_W[f] for f in ft & fe)
    t_only = sum(_W[f] for f in ft - fe)
    e_only = sum(_W[f] for f in fe - ft)
    return common / 10.0, t_only / 10.0, e_only / 10.0


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")][:MAXK]
    t = int(target)
    out = {}
    for k in range(MAXK):
        if k < len(ex):
            c, a, b = _contrast(t, ex[k])
            out[f"common_{k}"] = c
            out[f"tonly_{k}"] = a
            out[f"eonly_{k}"] = b
            out[f"dist_{k}"] = abs(t - ex[k]) / 10.0
            out[f"present_{k}"] = 1.0
        else:
            for nm in ("common", "tonly", "eonly", "dist", "present"):
                out[f"{nm}_{k}"] = 0.0
    out["log_n_examples"] = log(len(ex))
    return out


def _stack(name):
    return pt.stack([pm.Data(f"{name}_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)


with pm.Model() as model:
    common = _stack("common")
    tonly = _stack("tonly")
    eonly = _stack("eonly")
    dist = _stack("dist")
    present = _stack("present")
    log_n = pm.Data("log_n_examples", np.zeros(1, dtype="float64"))

    # Contrast-model weights: common features, target-only and example-only features.
    theta = pt.exp(pm.Normal("log_theta", mu=-1.0, sigma=1.5))
    a_target = pt.exp(pm.Normal("log_a_target", mu=-1.0, sigma=1.5))
    b_example = pt.exp(pm.Normal("log_b_example", mu=-1.0, sigma=1.5))
    c_mag = pt.exp(pm.Normal("log_c_mag", mu=-3.0, sigma=1.5))
    kappa = pm.Beta("kappa", alpha=2.0, beta=2.0)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pt.exp(pm.Normal("log_beta", mu=0.5, sigma=1.0))

    log_sim = theta * common - a_target * tonly - b_example * eonly - c_mag * dist
    log_sim = pt.switch(present > 0.5, log_sim, -50.0)
    pooled = pt.logsumexp(log_sim, axis=1) - kappa * log_n

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(alpha + beta * pooled), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
