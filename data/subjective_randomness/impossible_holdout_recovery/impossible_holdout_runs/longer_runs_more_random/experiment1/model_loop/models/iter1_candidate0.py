"""
People judge the randomness of a sequence by focusing solely on its absolute longest streak of identical outcomes. Instead of adjusting for the total sequence length, they perceive a sequence as more random simply by counting the raw number of consecutive identical outcomes in its longest run.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    
    # Free cognitive parameter for sensitivity/noise
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # The perceived randomness score is the absolute maximum run length
    score_a = max_run_a
    score_b = max_run_b
    
    # Probability of choosing left (A)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))
    
    # Numerical safety: clip probability strictly inside (0, 1)
    p_left_safe = pt.clip(p_left, 1e-6, 1 - 1e-6)
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
