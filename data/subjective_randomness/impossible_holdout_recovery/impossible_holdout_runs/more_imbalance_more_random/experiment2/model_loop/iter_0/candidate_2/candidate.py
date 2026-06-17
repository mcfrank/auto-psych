"""
People judge sequence randomness by looking for the longest streak of identical outcomes, perceiving sequences where the longest streak takes up a smaller proportion of the sequence length as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    
    # Cognitive parameter
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    # Evidence for A and B. 
    # Smaller proportion of sequence taken up by longest streak -> more random.
    score_a = -tau * max_run_norm_a
    score_b = -tau * max_run_norm_b
    
    # Probability of choosing A (left)
    p_left = pm.Deterministic("p_left", pt.clip(pm.math.sigmoid(score_a - score_b), 1e-6, 1 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
