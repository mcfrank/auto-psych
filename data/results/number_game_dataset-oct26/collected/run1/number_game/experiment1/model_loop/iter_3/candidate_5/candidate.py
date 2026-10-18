"""Salience-weighted shared-property rule matching.

A refinement of shared_property_rule_matching: people note which arithmetic
properties all the examples share (even, odd, multiple of 3/4/5/10, prime,
square, power of 2, a common final digit) and say "yes" to the degree that the
target has those properties, regardless of magnitude. The one change: each
kind of property has its own inferred salience, so "yes" follows the
salience-weighted share of the shared properties the target matches.
"""
import numpy as np
import pymc as pm

_PRIMES = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}
_POW2 = {2 ** k for k in range(0, 7)}
_KINDS = ["even", "odd", "m3", "m4", "m5", "m10", "prime", "square", "pow2", "end"]


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
    }


def compute_features(number_set, target):
    """Per property kind: is it shared by all examples, and does the target match it."""
    examples = [int(x) for x in number_set.split(",")]
    t = int(target)
    tp = _props(t)
    out = {}
    for k in _KINDS[:-1]:
        shared = all(_props(e)[k] for e in examples)
        out["shared_" + k] = float(shared)
        out["match_" + k] = float(shared and tp[k])
    end_shared = len({e % 10 for e in examples}) == 1
    out["shared_end"] = float(end_shared)
    out["match_end"] = float(end_shared and t % 10 == examples[0] % 10)
    return out


with pm.Model() as model:
    shared = pm.math.stack(
        [pm.Data("shared_" + k, np.zeros(1, dtype="float64")) for k in _KINDS], axis=1
    )
    match = pm.math.stack(
        [pm.Data("match_" + k, np.zeros(1, dtype="float64")) for k in _KINDS], axis=1
    )

    # Relative salience of each property kind (only ratios matter).
    salience = pm.Dirichlet("salience", a=2.0 * np.ones(len(_KINDS)))
    denom = pm.math.dot(shared, salience)
    numer = pm.math.dot(match, salience)
    any_shared = pm.math.gt(pm.math.sum(shared, axis=1), 0.5)
    rule_match = pm.math.switch(any_shared, numer / pm.math.maximum(denom, 1e-6) - 0.5, 0.0)

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    sharpness = pm.HalfNormal("sharpness", sigma=10.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + sharpness * rule_match))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
