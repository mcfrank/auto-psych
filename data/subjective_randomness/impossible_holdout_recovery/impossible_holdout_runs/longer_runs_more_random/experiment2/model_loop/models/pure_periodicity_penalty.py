"""People judge the randomness of a sequence strictly by penalizing its periodicity, perceiving sequences with fewer short, repeating patterns as more random."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    # Higher periodicity means more repeating patterns, which are penalized
    score_a = -tau * periodicity_a
    score_b = -tau * periodicity_b
    
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
