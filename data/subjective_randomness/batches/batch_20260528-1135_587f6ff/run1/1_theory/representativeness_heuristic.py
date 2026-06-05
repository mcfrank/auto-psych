# file: representativeness_heuristic.py
"""
A representativeness heuristic model where observers prefer sequences whose
proportion of heads is closer to 0.5.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: proportion of heads for sequences A and B
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter: softmax temperature
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Score is negative distance from 0.5 (closer to 0.5 is better/more representative)
    score_a = -pt.abs(p_a - 0.5)
    score_b = -pt.abs(p_b - 0.5)

    # Probability of choosing sequence A
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    # Observed responses
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
