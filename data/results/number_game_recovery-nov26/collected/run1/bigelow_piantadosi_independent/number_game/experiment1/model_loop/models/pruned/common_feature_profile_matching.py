"""Common-feature profile matching.

People judge membership by common-feature matching rather than probabilistic
concept induction: they notice which arithmetic and surface properties
(parity, multiples, powers, squares, primes, final digit, contained digits,
decade) every example shares, and say yes in proportion to how much of that
shared profile the target also has, rarer shared properties counting more.
Numerical proximity plays no role.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = np.arange(1, 101)


def _properties():
    props = []
    props.append(N % 2 == 1)
    props.append(N % 2 == 0)
    props.append(np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in N]))
    props.append(np.isin(N, [k * k for k in range(1, 11)]))
    props.append(np.isin(N, [k ** 3 for k in range(1, 5)]))
    for b in range(2, 11):
        props.append(np.isin(N, [b ** e for e in range(1, 7) if b ** e <= 100]))
    for m in range(3, 13):
        props.append(N % m == 0)
    for d in range(10):
        props.append(N % 10 == d)
    for d in range(10):
        props.append(np.array([str(d) in str(n) for n in N]))
    for t in range(10):
        props.append((N // 10) == t)
    return np.array(props, dtype=float)  # (P, 100)


PROPS = _properties()
LOG_SIZE = np.log(PROPS.sum(axis=1))


def prepare_observed(rows):
    shared = np.zeros((len(rows), PROPS.shape[0]))
    match = np.zeros_like(shared)
    for i, r in enumerate(rows):
        ex = [int(x) for x in r["number_set"].split(",")]
        t = int(r["target"])
        shared[i] = PROPS[:, [e - 1 for e in ex]].min(axis=1)
        match[i] = PROPS[:, t - 1]
    return {
        "shared": shared,
        "shared_match": shared * match,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    shared = pm.Data("shared", np.zeros((1, PROPS.shape[0])))
    shared_match = pm.Data("shared_match", np.zeros((1, PROPS.shape[0])))

    # Weight of a shared property falls with its size (distinctiveness).
    gamma = pm.HalfNormal("gamma", sigma=1.0)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=5.0)

    w = pt.exp(-gamma * pt.as_tensor_variable(LOG_SIZE))
    num = pt.dot(shared_match, w)
    den = pt.dot(shared, w) + 1e-9
    frac = num / den

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(intercept + slope * frac), 1e-6, 1 - 1e-6))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
