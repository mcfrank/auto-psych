"""Distinctive feature matching (Tversky-style common-feature similarity).

People do not entertain explicit hypotheses about the hidden program; they
notice which arithmetic and digit properties (parity, primality, squares,
cubes, powers, multiples of k, final digit, contained digits, decade) all the
examples share, and say "yes" in proportion to how much of that shared-feature
profile the target also shows, with rarer (more distinctive) shared properties
weighted more heavily than common ones.
"""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _properties():
    nums = range(1, 101)
    props = [
        {n for n in nums if n % 2 == 1},
        {n for n in nums if n % 2 == 0},
        {n for n in nums if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))},
        {n for n in nums if int(round(n ** 0.5)) ** 2 == n},
        {n for n in nums if int(round(n ** (1 / 3))) ** 3 == n},
    ]
    for b in range(2, 11):
        props.append({b ** k for k in range(1, 8) if b ** k <= 100})
    for k in range(3, 13):
        props.append({n for n in nums if n % k == 0})
    for d in range(10):
        props.append({n for n in nums if n % 10 == d})
    for d in range(10):
        props.append({n for n in nums if str(d) in str(n)})
    for lo in range(1, 101, 10):
        props.append(set(range(lo, lo + 10)))
    return props


_PROPS = _properties()
_INFO = np.array([math.log(100.0 / len(p)) for p in _PROPS])  # distinctiveness


def prepare_observed(rows):
    shared, has = [], []
    for r in rows:
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        s = [1.0 if all(e in p for e in ex) else 0.0 for p in _PROPS]
        shared.append(s)
        has.append([1.0 if t in p else 0.0 for p in _PROPS])
    return {
        "shared": np.array(shared, dtype="float64"),
        "target_has": np.array(has, dtype="float64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    shared = pm.Data("shared", np.zeros((1, len(_PROPS)), dtype="float64"))
    target_has = pm.Data("target_has", np.zeros((1, len(_PROPS)), dtype="float64"))

    # How strongly distinctiveness (rarity) of a shared property scales its weight.
    gamma = pm.HalfNormal("gamma", sigma=1.5)
    # Baseline log-odds of "yes" for a target matching none of the shared profile.
    alpha = pm.Normal("alpha", mu=-2.0, sigma=2.0)
    # Gain on the matched fraction of the shared-feature profile.
    beta = pm.HalfNormal("beta", sigma=5.0)

    w = pt.exp(gamma * pt.log(pt.as_tensor_variable(_INFO) + 1e-3))
    sw = shared * w[None, :]
    match = (sw * target_has).sum(axis=1) / (sw.sum(axis=1) + 1e-6)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(alpha + beta * match), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
