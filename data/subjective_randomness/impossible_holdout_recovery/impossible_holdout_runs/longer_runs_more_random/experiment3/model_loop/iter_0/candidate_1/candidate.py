"""
People judge the randomness of a sequence strictly by comparing its rate of alternations to a subjective ideal alternation rate. They perceive a sequence as more random the closer its alternation proportion is to this expected ideal, penalizing sequences that either alternate too much or too little.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    ideal_alts = pm.Beta("ideal_alts", alpha=5.0, beta=5.0)
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Subjective randomness is penalized by distance from the ideal alternation rate
    dist_a = pt.abs(p_alts_a - ideal_alts)
    dist_b = pt.abs(p_alts_b - ideal_alts)

    # p_left is higher when dist_a is smaller than dist_b
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dist_b - dist_a)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
