"""People judge the randomness of a sequence by comparing its absolute longest streak of identical outcomes to a subjective ideal absolute streak length, regardless of total sequence length."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    
    # Prior over ideal absolute streak length (e.g. 2, 3, 4)
    ideal_run = pm.HalfNormal("ideal_run", sigma=5.0)
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    dist_a = pt.abs(pt.cast(max_run_a, "float64") - ideal_run)
    dist_b = pt.abs(pt.cast(max_run_b, "float64") - ideal_run)
    
    score_a = -tau * dist_a
    score_b = -tau * dist_b
    
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
