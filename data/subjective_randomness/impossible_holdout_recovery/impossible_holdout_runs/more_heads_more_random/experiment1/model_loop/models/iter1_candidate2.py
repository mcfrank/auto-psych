"""
People evaluate the randomness of a sequence by calculating the excess of heads over tails, expecting random sequences to be heavily biased. They compute a randomness score equal to the number of heads minus the number of tails, perceiving sequences with a greater excess of heads as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    
    # Free cognitive parameter: temperature for the softmax/sigmoid
    tau = pm.HalfNormal("tau", sigma=1.0)
    
    # Tails = n - h
    # Excess of heads over tails = h - (n - h) = 2h - n
    score_a = 2 * h_a - n_a
    score_b = 2 * h_b - n_b
    
    # Calculate probability of choosing left (option A)
    # tau scales the difference in scores
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
