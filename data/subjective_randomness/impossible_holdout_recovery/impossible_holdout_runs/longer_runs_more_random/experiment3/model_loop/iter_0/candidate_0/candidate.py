"""
People judge the randomness of a sequence by comparing its maximum run proportion to a subjective ideal proportion, but they penalize deviations from this ideal using a squared distance, causing extreme deviations to seem disproportionately less random than minor ones.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    
    # Subjective ideal run proportion [0, 1]
    ideal_run = pm.Beta("ideal_run", alpha=2.0, beta=2.0)
    
    # Sensitivity to squared deviations
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    # Score is negative squared deviation: peaks at 0 when max_run_norm == ideal_run
    # Extreme deviations result in more negative scores (less random).
    score_a = -tau * pt.square(max_run_norm_a - ideal_run)
    score_b = -tau * pt.square(max_run_norm_b - ideal_run)
    
    # Probability of choosing left (A) is sigmoid of the score difference
    p_left = pm.Deterministic("p_left", pt.clip(pm.math.sigmoid(score_a - score_b), 1e-6, 1 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
