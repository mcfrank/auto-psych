# file: representativeness_heuristic.py
"""
Representativeness heuristic: observers prefer sequences whose head proportion is closer to 0.5.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: head proportions for the two sequences
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter: softmax temperature
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Distance of head proportion from 0.5
    dist_a = pt.abs(p_a - 0.5)
    dist_b = pt.abs(p_b - 0.5)

    # Smaller distance means more representative, so the evidence for A over B is dist_b - dist_a
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dist_b - dist_a)))

    # Observed responses
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
