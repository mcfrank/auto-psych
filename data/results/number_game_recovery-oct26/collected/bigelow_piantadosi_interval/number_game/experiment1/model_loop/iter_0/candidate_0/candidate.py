"""Shared-property matching (feature-overlap similarity, no size principle).

People note every property that all the examples have in common (parity,
final digit, tens decade, being a multiple, power, square, cube or prime,
lying within the examples' range) and endorse a target in proportion to how
many of those shared properties it also has, regardless of how many numbers
in 1-100 share them.
"""
import numpy as np
import pymc as pm


def _is_prime(n):
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n ** 0.5) + 1))


_SQUARES = {k * k for k in range(1, 11)}
_CUBES = {k ** 3 for k in range(1, 5)}
_POWERS = {b: {b ** e for e in range(1, 8) if b ** e <= 100} for b in range(2, 11)}


def _properties(n):
    props = {"even": n % 2 == 0, "odd": n % 2 == 1, "prime": _is_prime(n),
             "square": n in _SQUARES, "cube": n in _CUBES}
    for m in range(3, 13):
        props[f"mult_{m}"] = n % m == 0
    for b, s in _POWERS.items():
        props[f"pow_{b}"] = n in s
    for d in range(10):
        props[f"last_{d}"] = n % 10 == d
    for t in range(11):
        props[f"decade_{t}"] = n // 10 == t
    return props


def compute_features(number_set, target):
    """Fraction of the examples' shared properties that the target also has."""
    xs = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    ex_props = [_properties(x) for x in xs]
    t_props = _properties(t)
    shared = [k for k in t_props if all(p[k] for p in ex_props)]
    matched = sum(1 for k in shared if t_props[k])
    # The examples always share "lying within their own range".
    matched += 1 if min(xs) <= t <= max(xs) else 0
    return {"match_fraction": matched / (len(shared) + 1.0)}


with pm.Model() as model:
    match_fraction = pm.Data("match_fraction", np.zeros(1, dtype="float64"))

    # Baseline log-odds of "yes" when the target matches none of the shared
    # properties, and how strongly matching them all raises it.
    intercept = pm.Normal("intercept", mu=-1.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=5.0)

    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + slope * match_fraction))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
