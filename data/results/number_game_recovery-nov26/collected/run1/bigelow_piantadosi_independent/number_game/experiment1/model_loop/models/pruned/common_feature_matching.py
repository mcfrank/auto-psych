"""Common-feature matching.

People abstract the set of features that every example has in common (parity,
divisibility, being a square/cube/prime/power, final digit, decade) and say
"yes" to a target in proportion to how many of those shared features it also
has. No size principle: a shared feature is extended to any number that has
it, however far away; more examples matter only by shrinking the shared set.
"""
import numpy as np
import pymc as pm


def _features(n):
    sq = {k * k for k in range(1, 11)}
    cu = {k ** 3 for k in range(1, 5)}
    primes = {p for p in range(2, 101) if all(p % d for d in range(2, int(p ** 0.5) + 1))}
    f = set()
    f.add("odd" if n % 2 else "even")
    for m in range(3, 11):
        if n % m == 0:
            f.add(f"mult{m}")
    if n in sq:
        f.add("square")
    if n in cu:
        f.add("cube")
    if n in primes:
        f.add("prime")
    for b in (2, 3, 5):
        k = 1
        while k <= 100:
            if k == n:
                f.add(f"pow{b}")
            k *= b
    f.add(f"ends{n % 10}")
    f.add(f"decade{(n - 1) // 10}")
    return f


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    common = set.intersection(*[_features(x) for x in ex])
    t = _features(int(target))
    match = len(common & t) / len(common) if common else 0.0
    return {"common_match": float(match)}


with pm.Model() as model:
    common_match = pm.Data("common_match", np.zeros(1, dtype="float64"))
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=5.0)
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + slope * common_match)
    )
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
