"""
People judge sequence randomness purely based on the relative frequencies of outcomes: sequences with a smaller imbalance between the number of heads and tails are perceived as more random.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter for decision noise / sensitivity
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Imbalance is the absolute deviation of the proportion of heads from 0.5
    prop_a = h_a / n_a
    prop_b = h_b / n_b
    
    imbalance_a = pt.abs(prop_a - 0.5)
    imbalance_b = pt.abs(prop_b - 0.5)
    
    # Lower imbalance means higher perceived randomness
    score_a = -imbalance_a
    score_b = -imbalance_b
    
    # Calculate probability and clamp for numerical safety
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
