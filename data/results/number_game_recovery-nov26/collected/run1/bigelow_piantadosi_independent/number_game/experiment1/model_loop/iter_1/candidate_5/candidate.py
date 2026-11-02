"""Graded common-feature profile matching.

Refinement of common_feature_profile_matching: people match the target against
the arithmetic and surface properties of the examples (parity, multiples,
powers, squares, primes, final digit, contained digits, decade), rarer
properties counting more, but each property enters the profile in proportion
to the fraction of examples that have it (raised to a fitted exponent) instead
of only when all examples share it. A target matching a property held by only
some examples (e.g. a final digit) therefore still gains support.
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
    log_frac = np.zeros((len(rows), PROPS.shape[0]))
    present = np.zeros_like(log_frac)
    match = np.zeros_like(log_frac)
    for i, r in enumerate(rows):
        ex = [int(x) for x in r["number_set"].split(",")]
        t = int(r["target"])
        frac = PROPS[:, [e - 1 for e in ex]].mean(axis=1)
        present[i] = (frac > 0).astype(float)
        log_frac[i] = np.log(np.where(frac > 0, frac, 1.0))
        match[i] = PROPS[:, t - 1]
    return {
        "log_frac": log_frac,
        "present": present,
        "target_match": match,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    log_frac = pm.Data("log_frac", np.zeros((1, PROPS.shape[0])))
    present = pm.Data("present", np.zeros((1, PROPS.shape[0])))
    target_match = pm.Data("target_match", np.zeros((1, PROPS.shape[0])))

    gamma = pm.HalfNormal("gamma", sigma=1.0)       # distinctiveness weighting
    kappa = pm.HalfNormal("kappa", sigma=3.0)       # steepness of graded sharing
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=5.0)

    w = pt.exp(-gamma * pt.as_tensor_variable(LOG_SIZE))
    share = present * pt.exp(kappa * log_frac)       # fraction of examples ** kappa
    weighted = share * w
    num = pt.sum(weighted * target_match, axis=1)
    den = pt.sum(weighted, axis=1) + 1e-9
    frac = num / den

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(intercept + slope * frac), 1e-6, 1 - 1e-6))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
