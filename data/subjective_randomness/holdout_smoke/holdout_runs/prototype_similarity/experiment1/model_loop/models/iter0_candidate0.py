"""
People judge the randomness of a sequence by the proportion of the sequence made up of its longest continuous run, perceiving sequences where a single run occupies a larger fraction of the total length as less random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Sensitivity to the difference in normalized run length. 
    beta = pm.Normal("beta", mu=0.0, sigma=5.0)

    # Calculate probability of choosing sequence A.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (max_run_norm_a - max_run_norm_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
