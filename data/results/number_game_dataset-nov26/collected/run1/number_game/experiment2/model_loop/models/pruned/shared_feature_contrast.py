"""Shared-feature contrast (Tversky-style common-feature abstraction).

People do not compare the target to each example; they abstract the features
all the examples have in common (parity, multiples of k, final digit, decade,
squares, cubes, powers of two, primes), weight each shared feature by how
suspicious a coincidence it is (number of examples times log of how rare the
feature is in 1-100), and say yes in proportion to the weighted shared-feature
evidence the target satisfies minus the evidence it violates.
"""
from math import log

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _is_prime(x):
    return x > 1 and all(x % d for d in range(2, int(x ** 0.5) + 1))


_SQ = {i * i for i in range(1, 11)}
_CU = {i ** 3 for i in range(1, 5)}
_P2 = {2 ** i for i in range(0, 7)}
_FEATURES = []
_FEATURES.append(lambda x: x % 2 == 0)
_FEATURES.append(lambda x: x % 2 == 1)
for _k in range(3, 13):
    _FEATURES.append(lambda x, k=_k: x % k == 0)
for _d in range(10):
    _FEATURES.append(lambda x, d=_d: x % 10 == d)
for _t in range(11):
    _FEATURES.append(lambda x, t=_t: x // 10 == t)
_FEATURES.append(lambda x: x in _SQ)
_FEATURES.append(lambda x: x in _CU)
_FEATURES.append(lambda x: x in _P2)
_FEATURES.append(_is_prime)
_WEIGHT = [log(100.0 / max(1, sum(f(x) for x in range(1, 101)))) for f in _FEATURES]


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = len(ex)
    match = 0.0
    viol = 0.0
    for f, w in zip(_FEATURES, _WEIGHT):
        if all(f(e) for e in ex):
            if f(t):
                match += n * w
            else:
                viol += n * w
    return {"shared_match": match / 10.0, "shared_violation": viol / 10.0}


with pm.Model() as model:
    match = pm.Data("shared_match", np.zeros(1, dtype="float64"))
    viol = pm.Data("shared_violation", np.zeros(1, dtype="float64"))

    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    w_match = pm.HalfNormal("w_match", sigma=3.0)
    w_viol = pm.HalfNormal("w_viol", sigma=3.0)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(alpha + w_match * match - w_viol * viol), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
