"""
People rely on a simple "heads-equal-randomness" heuristic, where heads are viewed as independent random events and tails are viewed as non-random deterministic events. Consequently, the perceived randomness of a sequence strictly and monotonically increases with its proportion of heads.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    
    # Free cognitive parameter for choice stochasticity/sensitivity
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # Randomness score is directly proportional to the proportion of heads.
    # Score difference: p_a - p_b
    p_left_raw = pm.math.sigmoid(tau * (p_a - p_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
