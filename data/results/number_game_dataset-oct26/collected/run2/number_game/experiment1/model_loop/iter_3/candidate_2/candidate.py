"""Step-lattice progression inference.

People imagine the hidden program as counting in a fixed step: they infer the
largest step that carries every example to every other (the gcd of the
examples' differences) and reject targets not reachable from the examples in
whole steps. Belief in the stepping rule grows with how unlikely such
alignment is by chance: (n - 2) examples beyond the two that define the step,
each landing on the step lattice with chance 1/step, so belief = 1 - step^-(n-2).
With one or two examples (or step 1) there is no confirmed step and the answer
depends only on how close the target is to the examples.
"""
import math

import numpy as np
import pymc as pm


def compute_features(number_set, target):
    xs = [int(v) for v in number_set.split(",")]
    t = int(target)
    n = len(xs)
    step = 0
    for x in xs[1:]:
        step = math.gcd(step, abs(x - xs[0]))
    if n >= 3 and step >= 2:
        belief = 1.0 - float(step) ** (-(n - 2))
        off = 0.0 if (t - xs[0]) % step == 0 else 1.0
    else:
        belief = 0.0
        off = 0.0
    dist = min(abs(t - x) for x in xs) / 100.0
    return {"off_lattice_belief": off * belief, "near_dist": dist}


with pm.Model() as model:
    off_lattice_belief = pm.Data("off_lattice_belief", np.zeros(1, dtype="float64"))
    near_dist = pm.Data("near_dist", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    falloff = pm.HalfNormal("falloff", sigma=10.0)
    rule_strength = pm.HalfNormal("rule_strength", sigma=5.0)

    logit = intercept - falloff * near_dist - rule_strength * off_lattice_belief
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(logit))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
