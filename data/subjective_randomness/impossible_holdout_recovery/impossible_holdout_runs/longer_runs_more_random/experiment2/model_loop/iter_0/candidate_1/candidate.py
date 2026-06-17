"""
People judge the randomness of a sequence by comparing its alternation rate to their subjective ideal alternation rate. They perceive a sequence as more random the closer its alternation proportion is to this expected ideal, penalizing sequences that either alternate too rarely (streaky) or too frequently (perfectly alternating).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    
    # Free cognitive parameters
    ideal_alt = pm.Beta("ideal_alt", alpha=2.0, beta=2.0)
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # Mechanism: Absolute deviation from the ideal alternation rate
    # Negative sign because larger deviation means less random (lower score)
    score_a = -pt.abs(p_alts_a - ideal_alt)
    score_b = -pt.abs(p_alts_b - ideal_alt)
    
    # Sigmoid function for choice probability
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    
    # Numerical safety: clip probability strictly between 0 and 1
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
