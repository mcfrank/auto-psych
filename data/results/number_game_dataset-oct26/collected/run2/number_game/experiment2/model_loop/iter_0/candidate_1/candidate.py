"""Shared-property coincidence rule.

People read the examples for the arithmetic properties they all have in common
(same parity, same final digit, same decade, being multiples of the same
number) and treat each shared property as evidence for a rule in proportion to
how surprising it would be for that many examples to share it by coincidence
(k * -log base rate). They say "yes" to a target according to the
surprise-weighted balance of shared properties it has versus lacks, discounted
only by how far the target falls outside the range the examples span; no
similarity to individual examples is computed.
"""
import math

import numpy as np
import pymc as pm


def _properties(n):
    props = {("parity", n % 2): 0.5, ("digit", n % 10): 0.1, ("decade", min(n, 99) // 10): 0.1}
    for m in range(3, 11):
        if n % m == 0:
            props[("mult", m)] = 1.0 / m
    return props


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    k = len(ex)
    shared = _properties(ex[0])
    for x in ex[1:]:
        px = _properties(x)
        shared = {f: b for f, b in shared.items() if f in px}
    tp = _properties(t)
    signed = sum((1.0 if f in tp else -1.0) * (-k * math.log(b)) for f, b in shared.items())
    outside = max(0, min(ex) - t, t - max(ex)) / 100.0
    return {"rule_evidence": signed / 10.0, "range_outside": outside}


with pm.Model() as model:
    rule_evidence = pm.Data("rule_evidence", np.zeros(1))
    range_outside = pm.Data("range_outside", np.zeros(1))

    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    rule_gain = pm.HalfNormal("rule_gain", sigma=2.0)
    falloff = pm.HalfNormal("falloff", sigma=10.0)

    logit = bias + rule_gain * rule_evidence - falloff * range_outside
    p_yes = pm.Deterministic("p_yes", pm.math.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
