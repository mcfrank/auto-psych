"""Shared-property overlap without the size principle.

People note every salient number property that all the examples share
(parity, final digit, decade, contained digit, prime, square, cube,
power-of-k, multiple-of-k) and say "yes" more often the larger the fraction of
those shared properties the target also has, every shared property counting
equally. It departs most from size-principle Bayesian concept learning on
targets that have the examples' broad properties but not their narrowest one.
"""

import numpy as np
import pymc as pm

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
          67, 71, 73, 79, 83, 89, 97}
POWERS = {k: {k ** e for e in range(1, 8) if k ** e <= 100} for k in range(2, 11)}


def properties(n):
    """The set of salient properties number n has."""
    props = {"even" if n % 2 == 0 else "odd", f"last{n % 10}", f"decade{n // 10 if n < 100 else 9}"}
    props |= {f"has{d}" for d in set(str(n))}
    if n in PRIMES:
        props.add("prime")
    if round(n ** 0.5) ** 2 == n:
        props.add("square")
    if round(n ** (1 / 3)) ** 3 == n:
        props.add("cube")
    props |= {f"pow{k}" for k, s in POWERS.items() if n in s}
    props |= {f"mult{k}" for k in range(3, 13) if n % k == 0}
    return props


def compute_features(number_set, target):
    examples = [int(x) for x in str(number_set).split(",")]
    shared = set.intersection(*(properties(x) for x in examples))
    t = properties(int(target))
    frac = len(shared & t) / len(shared) if shared else 0.0
    return {"shared_overlap": float(frac)}


with pm.Model() as model:
    shared_overlap = pm.Data("shared_overlap", np.zeros(1, dtype="float64"))

    # Baseline log-odds of "yes" for a target with none of the shared properties,
    # and how strongly each additional fraction of matched properties raises it.
    intercept = pm.Normal("intercept", mu=-1.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=5.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * shared_overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
