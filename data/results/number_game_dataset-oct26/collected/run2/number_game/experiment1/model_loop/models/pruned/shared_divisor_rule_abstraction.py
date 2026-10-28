"""Shared-divisor rule abstraction.

People abstract a single explicit rule from the examples: "multiples of g",
where g is the greatest common divisor the examples share. They endorse
targets that are multiples of g and reject those that are not, trusting the
rule more the less likely the examples would share g by coincidence (evidence
n * log g: more examples, larger g). When the examples share no factor above
1, only the target's closeness to the examples matters.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    t = int(target)
    g = reduce(math.gcd, examples)
    if g >= 2:
        evidence = len(examples) * math.log(g) / math.log(100.0)
        sign = 1.0 if t % g == 0 else -1.0
        rule = sign * evidence
    else:
        rule = 0.0
    dist = min(abs(t - x) for x in examples) / 100.0
    return {"gcd_rule_evidence": rule, "nearest_distance": dist}


with pm.Model() as model:
    rule_ev = pm.Data("gcd_rule_evidence", np.zeros(1, dtype="float64"))
    dist = pm.Data("nearest_distance", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    rule_trust = pm.HalfNormal("rule_trust", sigma=5.0)
    falloff = pm.HalfNormal("falloff", sigma=10.0)

    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + rule_trust * rule_ev - falloff * dist)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
