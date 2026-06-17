"""
People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes. Because they believe true randomness naturally produces long streaks, they perceive a sequence as more random the longer its maximum run is relative to the total sequence length.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    
    # Free cognitive parameter for sensitivity/noise
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # The perceived randomness score is directly proportional to the normalized maximum run length
    score_a = max_run_norm_a
    score_b = max_run_norm_b
    
    # Probability of choosing left (A)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
