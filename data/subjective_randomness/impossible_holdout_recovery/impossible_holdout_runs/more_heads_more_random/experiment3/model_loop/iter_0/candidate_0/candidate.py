"""
People evaluate the randomness of a sequence based on the logarithm of the absolute number of heads it contains, reflecting a diminishing sensitivity to head count differences as the total number of heads increases, and their choices are subject to a constant lapse rate for random guessing.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Cognitive parameters
    tau = pm.HalfNormal("tau", sigma=5.0)
    lapse = pm.Beta("lapse", alpha=1.0, beta=9.0)

    # Use log(heads + 1) to avoid log(0) and represent diminishing sensitivity
    log_h_a = pt.log(pt.cast(h_a, "float64") + 1.0)
    log_h_b = pt.log(pt.cast(h_b, "float64") + 1.0)

    # Logistic choice probability with lapse rate
    p_heuristic = pm.math.sigmoid(tau * (log_h_a - log_h_b))
    p_left_raw = lapse / 2.0 + (1.0 - lapse) * p_heuristic
    
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
