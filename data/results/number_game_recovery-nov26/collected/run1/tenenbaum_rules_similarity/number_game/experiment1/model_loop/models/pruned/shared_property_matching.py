"""Shared-property matching.

People do not weigh competing concepts by how well they explain the examples;
instead they note every salient number property that all the examples share
(parity, final digit, tens digit, being a multiple or power of something,
prime, square, containing a digit) and say "yes" to a target in proportion to
how many of those shared properties it also has. A target is rejected to the
degree it breaks the examples' common properties, regardless of how close it
lies in magnitude to any example.

Feature: the fraction of the properties common to all examples that the
target lacks. "yes" = sigmoid(intercept - sensitivity * fraction_violated).
"""

import numpy as np
import pymc as pm


def _is_prime(n):
    return n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))


def _properties():
    props = [
        lambda n: n % 2 == 1,
        lambda n: n % 2 == 0,
        _is_prime,
        lambda n: int(round(n**0.5)) ** 2 == n,
        lambda n: n in (1, 8, 27, 64),
    ]
    props += [(lambda m: lambda n: n % m == 0)(m) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    props += [(lambda m: lambda n: n in [m**k for k in range(1, 8)])(m) for m in range(2, 11)]
    props += [(lambda d: lambda n: n % 10 == d)(d) for d in range(10)]
    props += [(lambda d: lambda n: str(d) in str(n))(d) for d in range(10)]
    props += [(lambda t: lambda n: min(n, 99) // 10 == t)(t) for t in range(10)]  # same decade
    return props


PROPS = _properties()
TABLE = np.array([[bool(p(n)) for p in PROPS] for n in range(1, 101)])  # (100, props)


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    shared = TABLE[[x - 1 for x in examples]].all(axis=0)
    n_shared = shared.sum()
    violated = (shared & ~TABLE[int(target) - 1]).sum()
    return {"frac_violated": float(violated / n_shared) if n_shared else 0.0}


with pm.Model() as model:
    frac_violated = pm.Data("frac_violated", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    sensitivity = pm.HalfNormal("sensitivity", sigma=5.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept - sensitivity * frac_violated))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
