"""
People judge sequence randomness by comparing the proportion of alternating outcomes to a subjective ideal alternation rate, perceiving sequences closer to this ideal as more random.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    
    ideal_rate = pm.Uniform("ideal_rate", lower=0.1, upper=0.9)
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    score_a = -pt.abs(p_alts_a - ideal_rate)
    score_b = -pt.abs(p_alts_b - ideal_rate)
    
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
