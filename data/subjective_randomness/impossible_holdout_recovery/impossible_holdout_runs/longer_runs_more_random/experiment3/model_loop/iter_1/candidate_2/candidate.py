"""
People judge the randomness of a sequence by comparing its longest streak proportion to a subjective ideal, but they evaluate deviations asymmetrically. Sequences that are excessively streaky (exceeding the ideal) are penalized at a different, typically steeper rate than sequences that are overly alternating (falling short of the ideal).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Cognitive parameters
    # The ideal maximum run proportion (expected to be somewhere around 0.3-0.5 for typical sequences)
    ideal_run = pm.Beta("ideal_run", alpha=2.0, beta=5.0)
    
    # Asymmetric sensitivity parameters
    slope_short = pm.HalfNormal("slope_short", sigma=10.0)
    slope_long = pm.HalfNormal("slope_long", sigma=10.0)

    # Difference from ideal
    diff_a = max_run_norm_a - ideal_run
    diff_b = max_run_norm_b - ideal_run

    # Asymmetric penalty: slope_long for positive diff (too streaky), slope_short for negative diff (too alternating)
    penalty_a = slope_long * pt.maximum(0, diff_a) + slope_short * pt.maximum(0, -diff_a)
    penalty_b = slope_long * pt.maximum(0, diff_b) + slope_short * pt.maximum(0, -diff_b)

    # Choice probability: higher penalty means lower perceived randomness
    # p_left = sigmoid(score_a - score_b) = sigmoid(-penalty_a - (-penalty_b)) = sigmoid(penalty_b - penalty_a)
    p_left_raw = pm.math.sigmoid(penalty_b - penalty_a)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
