"""Shared-property profile matching.

People note which arithmetic and digit properties (parity, primality, squares,
cubes, powers, multiples, final digit, contained digit) every example shares,
and say "yes" in proportion to how much of that shared-property profile the
target also has, rarer (more distinctive) properties counting more. There is
no interval / magnitude generalisation: a target adjacent to an example that
breaks the shared properties is usually rejected.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _properties():
    nums = np.arange(1, 101)
    props = [nums % 2 == 1, nums % 2 == 0]
    primes = np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in nums])
    props.append(primes)
    props.append(np.isin(nums, [k * k for k in range(1, 11)]))
    props.append(np.isin(nums, [k ** 3 for k in range(1, 5)]))
    for b in range(2, 11):
        props.append(np.isin(nums, [b ** k for k in range(1, 8)]))
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        props.append(nums % m == 0)
    for d in range(10):
        props.append(nums % 10 == d)
    for d in range(10):
        props.append(np.array([str(d) in str(n) for n in nums]))
    return np.array(props, dtype="float64")  # (F, 100)


PROPS = _properties()
LOG_SIZE = np.log(PROPS.sum(axis=1))


def prepare_observed(rows):
    shared, has = [], []
    for r in rows:
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        shared.append(PROPS[:, [e - 1 for e in ex]].min(axis=1))
        has.append(PROPS[:, t - 1])
    return {
        "shared": np.array(shared, dtype="float64"),
        "target_has": np.array(has, dtype="float64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


F = PROPS.shape[0]

with pm.Model() as model:
    shared = pm.Data("shared", np.zeros((1, F), dtype="float64"))
    target_has = pm.Data("target_has", np.zeros((1, F), dtype="float64"))

    # How much more distinctive (smaller) properties weigh.
    distinctiveness = pm.HalfNormal("distinctiveness", sigma=1.0)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    match_gain = pm.HalfNormal("match_gain", sigma=5.0)

    w = pt.exp(-distinctiveness * pt.as_tensor_variable(LOG_SIZE))  # (F,)
    total = (shared * w).sum(axis=1)
    matched = (shared * target_has * w).sum(axis=1)
    match = matched / (total + 1e-9)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + match_gain * match))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
