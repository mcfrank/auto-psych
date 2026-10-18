"""Shared-property rule matching plus range filling.

A refinement of shared_property_rule_matching: people say "yes" to the degree
that the target has the arithmetic properties all the examples share,
regardless of its distance from them, and in addition treat the span between
the smallest and largest example as covered by the program, so a target inside
that range gets an extra, inferred boost toward "yes".
"""
import numpy as np
import pymc as pm

_PRIMES = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}
_POW2 = {2 ** k for k in range(0, 7)}


def _props(n):
    return {
        "even": n % 2 == 0,
        "odd": n % 2 == 1,
        "m3": n % 3 == 0,
        "m4": n % 4 == 0,
        "m5": n % 5 == 0,
        "m10": n % 10 == 0,
        "prime": n in _PRIMES,
        "square": n in _SQUARES,
        "pow2": n in _POW2,
        "end%d" % (n % 10): True,
    }


def compute_features(number_set, target):
    """Rule match (as in the incumbent) and whether the target lies within the examples' range."""
    examples = [int(x) for x in number_set.split(",")]
    t = int(target)
    shared = None
    for e in examples:
        p = {k for k, v in _props(e).items() if v}
        shared = p if shared is None else shared & p
    tp = {k for k, v in _props(t).items() if v}
    match = 0.0 if not shared else len(shared & tp) / len(shared) - 0.5
    in_range = 1.0 if min(examples) <= t <= max(examples) else 0.0
    return {"rule_match": match, "in_range": in_range}


with pm.Model() as model:
    rule_match = pm.Data("rule_match", np.zeros(1, dtype="float64"))
    in_range = pm.Data("in_range", np.zeros(1, dtype="float64"))

    # Baseline willingness to say yes, how sharply rule matching drives it,
    # and the boost for lying within the examples' span.
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    sharpness = pm.HalfNormal("sharpness", sigma=10.0)
    range_boost = pm.Normal("range_boost", mu=0.0, sigma=2.0)

    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + sharpness * rule_match + range_boost * in_range)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
