"""
People judge the randomness of a sequence strictly by the absolute number of heads it contains, but their penalty for heads grows quadratically. Each additional head decreases the perceived randomness more than the previous one, meaning the perceived difference in randomness between 8 and 9 heads is much larger than between 1 and 2.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Quadratic penalty for number of heads
    score_a = - pt.cast(h_a ** 2, "float64")
    score_b = - pt.cast(h_b ** 2, "float64")

    # Difference in scores
    diff = score_a - score_b
    
    # Sigmoid to get probability, clipped for numerical safety
    p_raw = pm.math.sigmoid(tau * diff)
    p_left = pm.Deterministic("p_left", pt.clip(p_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
