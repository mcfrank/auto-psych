"""People judge the randomness of a sequence by comparing its longest streak of identical outcomes to their subjective ideal streak proportion. They perceive a sequence as more random the closer its maximum run proportion is to this expected ideal length, penalizing streaks that are either suspiciously short or excessively long."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    
    ideal_run_norm = pm.Beta("ideal_run_norm", alpha=2.0, beta=5.0)
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    dist_a = pt.abs(max_run_norm_a - ideal_run_norm)
    dist_b = pt.abs(max_run_norm_b - ideal_run_norm)
    
    score_a = -tau * dist_a
    score_b = -tau * dist_b
    
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
