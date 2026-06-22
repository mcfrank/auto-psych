"""
People evaluate the randomness of a sequence based solely on the proportion of the sequence occupied by its longest contiguous streak of identical outcomes, penalizing sequences where a single streak dominates the sequence length.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs 
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter for the penalty weight
    penalty_weight = pm.HalfNormal("penalty_weight", sigma=5.0)

    # Optional side bias
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # The randomness score is just the negative penalty on the normalized max run length
    # A larger max run proportion means a larger penalty (more negative score)
    score_a = -penalty_weight * max_run_norm_a
    score_b = -penalty_weight * max_run_norm_b

    p_left_raw = pm.math.sigmoid(score_a - score_b + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
