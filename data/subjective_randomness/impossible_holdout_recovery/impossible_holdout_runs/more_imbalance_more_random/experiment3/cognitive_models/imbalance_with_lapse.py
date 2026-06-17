"""People judge sequence randomness based purely on outcome frequencies, perceiving sequences with a greater proportion imbalance as more random, but their judgments are moderated by a constant lapse rate representing random guessing."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=10.0)
    lapse = pm.Beta("lapse", alpha=1.0, beta=9.0) # Prior expecting a low lapse rate (mean 0.1)

    # Score sequences based on imbalance
    score_a = imbalance_a
    score_b = imbalance_b
    
    # Base probability of choosing left based on subjective score difference
    p_choice = pm.math.sigmoid(tau * (score_a - score_b))
    
    # Apply lapse rate: guess randomly with probability 'lapse'
    p_left_raw = (1.0 - lapse) * p_choice + lapse * 0.5
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
