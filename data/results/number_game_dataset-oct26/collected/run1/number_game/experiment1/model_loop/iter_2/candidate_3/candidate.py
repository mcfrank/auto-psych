"""Rule matching with range extension.

A refinement of shared_property_rule_matching: people say "yes" to the degree
that the target has the arithmetic properties all the examples share, and in
addition treat the examples' span (min to max) as produced by the program, so
a target inside the span gets an inferred boost that fades smoothly with its
distance beyond the span's edges (for one example, distance from it). The one
change is this range-membership component grafted from
range_extension_by_spacing.
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
    """Rule match (as in the incumbent) and distance beyond the examples' span."""
    examples = [int(x) for x in number_set.split(",")]
    t = int(target)
    shared = None
    for e in examples:
        p = {k for k, v in _props(e).items() if v}
        shared = p if shared is None else shared & p
    tp = {k for k, v in _props(t).items() if v}
    rule_match = 0.0 if not shared else len(shared & tp) / len(shared) - 0.5
    lo, hi = min(examples), max(examples)
    beyond = max(lo - t, t - hi, 0) / 100.0
    return {"rule_match": rule_match, "beyond_range": float(beyond)}


with pm.Model() as model:
    rule_match = pm.Data("rule_match", np.zeros(1, dtype="float64"))
    beyond_range = pm.Data("beyond_range", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    sharpness = pm.HalfNormal("sharpness", sigma=10.0)
    # Boost for being within the examples' span, and how far past its edges it extends.
    range_boost = pm.HalfNormal("range_boost", sigma=2.0)
    range_width = pm.LogNormal("range_width", mu=np.log(0.1), sigma=1.0)

    in_range = pm.math.exp(-beyond_range / range_width)
    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(intercept + sharpness * rule_match + range_boost * (in_range - 0.5)),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
