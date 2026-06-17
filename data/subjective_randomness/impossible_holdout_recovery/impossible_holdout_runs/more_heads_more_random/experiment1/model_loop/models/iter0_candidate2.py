"""
People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains. They apply a simple additive heuristic where each additional head linearly increases the perceived randomness score, disregarding the sequence length and outcome order. When comparing two sequences, they are more likely to choose the sequence with the higher total head count as the more random one.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter: sensitivity to the difference in head counts
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Perceived randomness is simply proportional to the absolute number of heads
    # The probability of choosing sequence A (left) increases with its relative head count advantage
    p_left = pm.Deterministic(
        "p_left", 
        pm.math.sigmoid(tau * (h_a - h_b))
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
