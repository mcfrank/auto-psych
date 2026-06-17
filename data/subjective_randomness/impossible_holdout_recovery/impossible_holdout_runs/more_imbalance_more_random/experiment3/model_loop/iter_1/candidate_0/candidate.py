"""
People judge sequence randomness based purely on outcome frequencies, paradoxically perceiving greater proportion imbalance as more random, with the psychophysical scaling of this imbalance governed by a free power-law exponent.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    
    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=5.0)
    gamma = pm.LogNormal("gamma", mu=0.0, sigma=1.0)
    
    # Power-law functional form for the perception of imbalance.
    # A small epsilon is added to avoid numerical issues.
    score_a = pt.power(imbalance_a + 1e-6, gamma)
    score_b = pt.power(imbalance_b + 1e-6, gamma)
    
    # Probability of choosing sequence A (left) as more random
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
