"""
People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. Their trial-by-trial comparison follows a probit function, reflecting normally distributed evaluation noise, but their final choices incorporate a baseline rate of random guessing due to occasional attentional lapses.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    
    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=5.0)
    lapse = pm.Beta("lapse", alpha=1.0, beta=9.0)
    
    # Difference in heads (prefer fewer, so positive if h_b > h_a)
    diff = pt.cast(h_b - h_a, "float64")
    
    # Probit link function for the core cognitive process
    core_p_left = pm.math.invprobit(tau * diff)
    
    # Mix with random guessing (0.5) due to attentional lapses
    p_left = pm.Deterministic("p_left", lapse * 0.5 + (1.0 - lapse) * core_p_left)
    
    # Ensure numerical safety
    p_left_safe = pt.clip(p_left, 1e-6, 1 - 1e-6)
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
