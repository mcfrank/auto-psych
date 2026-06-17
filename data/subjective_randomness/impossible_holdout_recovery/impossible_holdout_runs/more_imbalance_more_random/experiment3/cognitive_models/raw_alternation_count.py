"""
People perceive sequences with a higher raw count of alternations as more random, regardless of sequence length.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    score_a = pt.cast(alts_a, "float64")
    score_b = pt.cast(alts_b, "float64")
    
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
