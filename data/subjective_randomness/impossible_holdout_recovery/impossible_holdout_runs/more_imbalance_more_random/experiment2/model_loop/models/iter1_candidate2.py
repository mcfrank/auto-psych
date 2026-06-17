"""
People judge sequence randomness based on outcome frequencies, paradoxically perceiving sequences with a greater proportion imbalance as more random, but their judgments are moderated by a constant lapse rate representing random guessing.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    # Lapse rate: probability of making a random guess
    # Beta prior favoring low but non-zero lapse rates
    epsilon = pm.Beta("epsilon", alpha=1.0, beta=9.0)

    # Core cognitive mechanism: difference in imbalance
    score_a = imbalance_a
    score_b = imbalance_b
    
    # Base probability of choosing A from the cognitive mechanism
    p_base = pm.math.sigmoid(tau * (score_a - score_b))
    
    # Final probability mixing the cognitive mechanism with random guessing (0.5)
    p_left_raw = (1.0 - epsilon) * p_base + epsilon * 0.5
    
    # Numerical safety clamp
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
