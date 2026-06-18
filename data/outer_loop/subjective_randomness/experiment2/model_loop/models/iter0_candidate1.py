"""
Observers evaluate the randomness of a sequence based on its longest unbroken streak of identical outcomes. They use the length of this maximum run as a heuristic for non-randomness, judging sequences with longer streaks as less random.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    
    # Sensitivity to the maximum run length
    beta = pm.HalfNormal("beta", sigma=2.0)
    
    # Score decreases as the maximum run increases
    score_a = -beta * max_run_a
    score_b = -beta * max_run_b
    
    # Sigmoid to get probability, clamped for numerical stability
    p_raw = pm.math.sigmoid(score_a - score_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_raw, 1e-6, 1 - 1e-6))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
