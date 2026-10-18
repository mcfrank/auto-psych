"""Shared-property rule matching.

People ignore magnitude and similarity and judge by rule matching: they note
which arithmetic properties all the examples have in common (even, odd,
multiple of 3/4/5/10, shared final digit, prime, square, power of 2) and say
"yes" to the degree that the target has those same properties, regardless of
its distance from the examples.
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
    """Fraction of the examples' shared properties the target also has (centred; 0 if none shared)."""
    examples = [int(x) for x in number_set.split(",")]
    shared = None
    for e in examples:
        p = {k for k, v in _props(e).items() if v}
        shared = p if shared is None else shared & p
    t = {k for k, v in _props(int(target)).items() if v}
    if not shared:
        return {"rule_match": 0.0}
    return {"rule_match": len(shared & t) / len(shared) - 0.5}


with pm.Model() as model:
    rule_match = pm.Data("rule_match", np.zeros(1, dtype="float64"))

    # Baseline willingness to say yes, and how sharply rule matching drives it.
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    sharpness = pm.HalfNormal("sharpness", sigma=10.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + sharpness * rule_match))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
