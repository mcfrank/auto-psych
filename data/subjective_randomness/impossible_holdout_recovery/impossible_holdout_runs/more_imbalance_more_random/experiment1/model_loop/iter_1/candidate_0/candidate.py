"""
People judge sequence randomness based purely on outcome frequencies, paradoxically perceiving sequences with a greater imbalance between the number of heads and tails as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter for sensitivity
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Higher imbalance means higher perceived randomness
    score_a = imbalance_a
    score_b = imbalance_b

    # Calculate probability and clamp for numerical safety
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
