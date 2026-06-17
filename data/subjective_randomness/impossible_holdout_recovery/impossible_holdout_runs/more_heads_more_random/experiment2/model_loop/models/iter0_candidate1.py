"""
People evaluate the randomness of a sequence strictly based on the proportion of heads it contains, perceiving sequences with a higher proportion of heads as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Probability of choosing left (A)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (p_a - p_b)))
    
    # Clip to ensure numerical stability
    p_left_safe = pt.clip(p_left, 1e-6, 1 - 1e-6)

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
