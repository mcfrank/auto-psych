# file: proportion_heuristic.py
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: head proportions for the two sequences
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter: softmax temperature
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Distance to 0.5 (smaller distance means more representative of a fair coin)
    dist_a = pt.abs(p_a - 0.5)
    dist_b = pt.abs(p_b - 0.5)

    # Value is negative distance (closer to 0.5 is better)
    V_a = -dist_a
    V_b = -dist_b

    # Probability of choosing sequence A
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (V_a - V_b)))

    # Observed responses
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
