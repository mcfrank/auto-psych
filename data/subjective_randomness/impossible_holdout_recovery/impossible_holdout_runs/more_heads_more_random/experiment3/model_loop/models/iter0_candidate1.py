"""
People evaluate the randomness of a sequence primarily based on the proportion of heads it contains relative to its total length, judging sequences with a higher fraction of heads to be more random, and their choices are subject to a constant lapse rate representing occasional random guessing.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    
    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=10.0)
    lapse = pm.Beta("lapse", alpha=1.0, beta=9.0)
    
    # Core mechanism: compare proportion of heads
    p_core = pm.math.sigmoid(tau * (p_a - p_b))
    
    # Apply lapse rate for random guessing
    p_left_val = (lapse / 2.0) + (1.0 - lapse) * p_core
    
    # Numerically safe probability
    p_left = pm.Deterministic("p_left", pt.clip(p_left_val, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
