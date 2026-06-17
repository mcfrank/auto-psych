import numpy as np
import pymc as pm
import pytensor.tensor as pt

"""
People judge sequence randomness based purely on the alternation rate heuristic: they compare the proportion of alternating outcomes in the sequence to a subjective ideal alternation rate. Sequences whose alternation rate is closer to this subjective ideal are perceived as more random.
"""

with pm.Model() as model:
    # Stimulus inputs
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    
    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=10.0)
    ideal_alts = pm.Beta("ideal_alts", alpha=2.0, beta=2.0)
    
    # Randomness score: negative absolute deviation from the ideal alternation rate
    score_a = -pt.abs(p_alts_a - ideal_alts)
    score_b = -pt.abs(p_alts_b - ideal_alts)
    
    # Probability of choosing sequence A (left)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
