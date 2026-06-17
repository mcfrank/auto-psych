"""
People judge sequence randomness based purely on the relative length of the longest streak of identical outcomes, perceiving sequences with a larger normalized maximum run as less random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    
    # Cognitive parameter: effect of normalized maximum run length on perceived randomness.
    # Positive weight means larger max_run_norm increases p_left.
    # The hypothesis predicts sequences with larger normalized maximum runs are perceived as *less* random,
    # so we expect a negative value, but we let the prior cover both directions.
    weight = pm.Normal("weight", mu=0.0, sigma=10.0)
    
    # Randomness value for each sequence
    v_a = weight * max_run_norm_a
    v_b = weight * max_run_norm_b
    
    # Probability of choosing sequence A (left)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(v_a - v_b))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
