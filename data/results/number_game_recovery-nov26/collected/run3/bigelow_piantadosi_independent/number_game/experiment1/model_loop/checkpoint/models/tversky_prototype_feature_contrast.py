"""Tversky contrast-model prototype matching.

People judge membership by feature matching rather than by Bayesian inference
over concepts: they abstract a prototype made of every property all the
examples share (parity, divisibility, being a square/cube/prime/power, last
digit, digits contained, decade, lying within the examples' range), and say
"yes" the more of those shared properties the target has, penalised for each
shared property it lacks and, more weakly, for each property it has that the
prototype lacks. There is no size principle: a broad shared property counts as
much as a narrow one.
"""
import numpy as np
import pymc as pm


def _is_prime(n):
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n ** 0.5) + 1))


def _features(n, lo, hi):
    f = set()
    f.add("odd" if n % 2 else "even")
    for k in range(3, 13):
        if n % k == 0:
            f.add(f"mult{k}")
    if round(n ** 0.5) ** 2 == n:
        f.add("square")
    if round(n ** (1 / 3)) ** 3 == n:
        f.add("cube")
    if _is_prime(n):
        f.add("prime")
    for b in range(2, 11):
        p = 1
        while p < n:
            p *= b
        if p == n:
            f.add(f"pow{b}")
    f.add(f"end{n % 10}")
    for d in set(str(n)):
        f.add(f"has{d}")
    f.add(f"decade{min(n // 10, 9)}")
    if lo <= n <= hi:
        f.add("in_range")
    return f


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    lo, hi = min(ex), max(ex)
    proto = set.intersection(*[_features(x, lo, hi) for x in ex])
    tf = _features(t, lo, hi)
    return {
        "n_common": float(len(proto & tf)),
        "n_missing": float(len(proto - tf)),
        "n_extra": float(len(tf - proto)),
    }


with pm.Model() as model:
    n_common = pm.Data("n_common", np.zeros(1, dtype="float64"))
    n_missing = pm.Data("n_missing", np.zeros(1, dtype="float64"))
    n_extra = pm.Data("n_extra", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    theta = pm.HalfNormal("theta", sigma=2.0)   # weight of shared features
    alpha = pm.HalfNormal("alpha", sigma=2.0)   # penalty: prototype features target lacks
    beta = pm.HalfNormal("beta", sigma=1.0)     # penalty: target features prototype lacks

    logit = intercept + theta * n_common - alpha * n_missing - beta * n_extra
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(logit))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
