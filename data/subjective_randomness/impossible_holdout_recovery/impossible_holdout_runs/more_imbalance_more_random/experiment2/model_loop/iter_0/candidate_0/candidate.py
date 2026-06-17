"""
People judge sequence randomness based on outcome frequencies, with perceived randomness growing quadratically as the proportion of heads and tails deviates further from a balanced distribution.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter
    weight = pm.Normal("weight", mu=0.0, sigma=10.0)

    # Quadratic deviation from balanced proportion (0.5)
    dev_a = pt.sqr(p_a - 0.5)
    dev_b = pt.sqr(p_b - 0.5)

    # Compute probability of choosing left (A)
    # If weight is positive, higher deviation means more likely to be chosen as random.
    score_a = weight * dev_a
    score_b = weight * dev_b
    
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
