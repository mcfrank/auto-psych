"""
People judge a sequence as more random when its alternation rate is closer to
a subjective ideal. When comparing two sequences, they prefer the one whose
alternation rate deviates less from this internal standard, treating both too
much streakiness and too much regularity as evidence of non-randomness. The
ideal rate is a free parameter inferred from the data.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    ideal_rate = pm.Beta("ideal_rate", alpha=5.0, beta=5.0)
    tau = pm.HalfNormal("tau", sigma=2.0)

    dev_a = pt.abs(p_alts_a - ideal_rate)
    dev_b = pt.abs(p_alts_b - ideal_rate)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dev_b - dev_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
