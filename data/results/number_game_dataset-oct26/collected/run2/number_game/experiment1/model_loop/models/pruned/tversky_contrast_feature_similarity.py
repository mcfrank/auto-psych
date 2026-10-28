"""Tversky contrast-model featural similarity.

People judge a target by Tversky-style featural similarity to the examples:
each number is represented as a set of salient arithmetic features (odd/even,
multiple of 3-10, square, cube, prime, power of two, final digit, decade), and
a target's similarity to an example rises with their common features and falls
with each one's distinctive features, weighted asymmetrically. "Yes" rises with
the target's average featural similarity to the examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
_PRIMES = {p for p in range(2, 101) if all(p % d for d in range(2, int(p ** 0.5) + 1))}
_SQUARES = {i * i for i in range(1, 11)}
_CUBES = {i ** 3 for i in range(1, 5)}
_POW2 = {2 ** i for i in range(0, 7)}


def _features(x):
    f = {"even" if x % 2 == 0 else "odd", f"last{x % 10}", f"decade{min(x, 99) // 10}"}
    for m in range(3, 11):
        if x % m == 0:
            f.add(f"mult{m}")
    if x in _SQUARES:
        f.add("square")
    if x in _CUBES:
        f.add("cube")
    if x in _PRIMES:
        f.add("prime")
    if x in _POW2:
        f.add("pow2")
    return f


def prepare_observed(rows):
    n = len(rows)
    common = np.zeros((n, MAXK))
    dist_t = np.zeros((n, MAXK))
    dist_e = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")][:MAXK]
        ft = _features(int(row["target"]))
        for i, x in enumerate(ex):
            fe = _features(x)
            common[r, i] = len(ft & fe)
            dist_t[r, i] = len(ft - fe)
            dist_e[r, i] = len(fe - ft)
            mask[r, i] = 1.0
        rating[r] = int(float(row["rating"]))
    return {"common": common, "dist_t": dist_t, "dist_e": dist_e,
            "ex_mask": mask, "rating": rating}


with pm.Model() as model:
    common = pm.Data("common", np.zeros((1, MAXK)))
    dist_t = pm.Data("dist_t", np.zeros((1, MAXK)))
    dist_e = pm.Data("dist_e", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))

    # Contrast-model weights: common features, target-only, example-only.
    theta = pm.HalfNormal("theta", sigma=1.0)
    alpha = pm.HalfNormal("alpha", sigma=1.0)
    beta = pm.HalfNormal("beta", sigma=1.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=3.0)

    contrast = theta * common - alpha * dist_t - beta * dist_e
    sim = pt.exp(pt.clip(contrast, -30.0, 10.0))
    familiarity = pt.sum(ex_mask * sim, axis=1) / pt.sum(ex_mask, axis=1)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + gain * familiarity), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
