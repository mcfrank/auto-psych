"""
People judge the randomness of a sequence by focusing solely on the absolute length of its longest streak of identical outcomes. Because they believe true randomness naturally produces long streaks, they perceive a sequence as more random the longer its absolute maximum run is, completely ignoring the total sequence length.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    
    # Free cognitive parameter for sensitivity
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # The perceived randomness score is the absolute maximum run length
    score_a = pt.cast(max_run_a, "float64")
    score_b = pt.cast(max_run_b, "float64")
    
    # Probability of choosing left (A) is sigmoid of score difference, clamped for safety
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
