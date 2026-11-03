"""Divisor-overlap similarity.

People judge a target by its arithmetic similarity to the examples rather than
by magnitude: two numbers are similar to the extent that they share divisors
(Jaccard overlap of their sets of divisors greater than one), and the
willingness to say "yes" rises smoothly with the target's average
divisor-overlap similarity to the examples. Even numbers all share the divisor
2 while odd numbers share no common divisor, so even sets generalize by parity
far more than odd sets do.
"""
import numpy as np
import pymc as pm


def _divisors(n):
    return {d for d in range(2, n + 1) if n % d == 0}


def _sim(a, b):
    if a == b:
        return 1.0
    da, db = _divisors(a), _divisors(b)
    union = da | db
    if not union:
        return 0.0
    return len(da & db) / len(union)


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    t = int(target)
    return {"divisor_similarity": float(np.mean([_sim(t, x) for x in examples]))}


with pm.Model() as model:
    divisor_similarity = pm.Data("divisor_similarity", np.zeros(1, dtype="float64"))

    # Baseline log-odds of "yes" for a target sharing no divisors with any example,
    # and the steepness of the generalization gradient over divisor overlap.
    intercept = pm.Normal("intercept", mu=-1.0, sigma=2.0)
    gradient = pm.HalfNormal("gradient", sigma=10.0)

    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + gradient * divisor_similarity)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
