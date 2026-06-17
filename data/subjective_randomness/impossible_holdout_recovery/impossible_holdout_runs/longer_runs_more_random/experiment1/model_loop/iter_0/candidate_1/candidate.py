"""
People judge the randomness of a sequence solely by evaluating its alternation rate. Influenced by the Gambler's Fallacy, they expect random sequences to self-correct and alternate more frequently than chance, so they perceive a sequence as more random the closer its alternation rate is to their subjective ideal rate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — alternation rates for sequences A and B
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    
    # Cognitive parameters
    # The subjective ideal alternation rate.
    ideal_alts = pm.Beta("ideal_alts", alpha=5.0, beta=5.0) 
    
    # Sensitivity parameter for the choice function
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    # Distance from ideal alternation rate
    dist_a = pt.abs(p_alts_a - ideal_alts)
    dist_b = pt.abs(p_alts_b - ideal_alts)
    
    # People prefer sequences with a smaller distance to the ideal alternation rate
    score_a = -dist_a
    score_b = -dist_b
    
    # Probability of choosing sequence A (left)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))
    
    # Numerical safety: clip probability strictly inside (0, 1)
    p_left_safe = pt.clip(p_left, 1e-6, 1 - 1e-6)
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
