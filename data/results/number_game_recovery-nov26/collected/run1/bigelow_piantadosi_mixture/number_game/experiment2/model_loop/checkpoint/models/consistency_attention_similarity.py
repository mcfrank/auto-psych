"""Consistency-attention similarity.

People judge a target by its similarity to the examples, attending to each
dimension of number similarity (magnitude band, last digit, parity,
divisibility, square, cube, power of 2, prime) in proportion to how
surprisingly consistently the examples agree on it: a property all examples
share gets attention equal to its surprise, -n * log(base rate in 1-100).
The target's psychological distance is the summed attention on the shared
properties it lacks plus its number-line distance from the nearest example;
the log-odds of "yes" falls linearly along this distance.
"""
import math

import numpy as np
import pymc as pm

N = np.arange(1, 101)
_SQ = {i * i for i in range(1, 11)}
_CU = {i ** 3 for i in range(1, 5)}
_P2 = {2 ** i for i in range(0, 7)}
_PR = {n for n in range(2, 101) if all(n % k for k in range(2, int(n ** 0.5) + 1))}

DIMS = [
    lambda n: n % 2,
    lambda n: n % 10,
    lambda n: min(n // 10, 9),
    lambda n: int(n % 3 == 0),
    lambda n: int(n % 4 == 0),
    lambda n: int(n % 5 == 0),
    lambda n: int(n in _SQ),
    lambda n: int(n in _CU),
    lambda n: int(n in _P2),
    lambda n: int(n in _PR),
]


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    feat = 0.0
    for f in DIMS:
        vals = {f(e) for e in ex}
        if len(vals) != 1:
            continue
        v = vals.pop()
        if f(t) == v:
            continue
        base = sum(1 for n in N if f(int(n)) == v) / 100.0
        feat += -len(ex) * math.log(base)
    dmag = min(abs(t - e) for e in ex) / 10.0
    return {"feat_dist": feat / 10.0, "mag_dist": dmag}


with pm.Model() as model:
    feat_dist = pm.Data("feat_dist", np.zeros(1, dtype="float64"))
    mag_dist = pm.Data("mag_dist", np.zeros(1, dtype="float64"))

    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    w_feat = pm.HalfNormal("w_feat", sigma=3.0)
    w_mag = pm.HalfNormal("w_mag", sigma=3.0)

    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(bias - w_feat * feat_dist - w_mag * mag_dist)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
