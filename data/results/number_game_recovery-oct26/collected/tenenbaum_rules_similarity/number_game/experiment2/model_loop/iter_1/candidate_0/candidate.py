"""Common-feature contrast similarity.

People judge a target by its similarity to the example set as a whole, not by
inducing a concept: similarity is the informativeness-weighted share of the
properties that all examples have in common (parity, multiples, final digit,
contained digit, powers, primes, squares, cubes, decade) that the target also
has, together with its closeness in magnitude to the nearest example on a
logarithmic distance scale. P(yes) rises smoothly with this similarity.
"""
import math

import numpy as np
import pymc as pm


def _properties():
    props = []
    for b in range(2, 11):
        s, v = set(), b
        while v <= 100:
            s.add(v)
            v *= b
        props.append(s)
    for m in range(3, 13):
        props.append(set(range(m, 101, m)))
    for d in range(10):
        props.append({n for n in range(1, 101) if n % 10 == d})
    for d in range(1, 10):
        props.append({n for n in range(1, 101) if str(d) in str(n)})
    props.append({n for n in range(2, 101) if all(n % k for k in range(2, int(n ** 0.5) + 1))})
    props.append(set(range(2, 101, 2)))
    props.append(set(range(1, 101, 2)))
    props.append({k * k for k in range(1, 11)})
    props.append({k ** 3 for k in range(1, 5)})
    for dec in range(10):
        props.append({n for n in range(1, 101) if (n - 1) // 10 == dec})
    return [(p, math.log(100.0 / len(p))) for p in props]


_PROPS = _properties()


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    common = [(p, w) for p, w in _PROPS if all(x in p for x in ex)]
    tot = sum(w for _, w in common)
    match = sum(w for p, w in common if t in p) / tot if tot > 0 else 0.0
    d = min(abs(t - x) for x in ex)
    return {"feature_match": float(match), "log_distance": float(math.log1p(d))}


with pm.Model() as model:
    feature_match = pm.Data("feature_match", np.zeros(1, dtype="float64"))
    log_distance = pm.Data("log_distance", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    feature_weight = pm.HalfNormal("feature_weight", sigma=5.0)
    distance_falloff = pm.HalfNormal("distance_falloff", sigma=2.0)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(intercept + feature_weight * feature_match - distance_falloff * log_distance),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
