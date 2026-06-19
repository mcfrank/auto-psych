"""
People judge a sequence as less random if it contains periodic, repeating patterns. When comparing two sequences, they prefer the one with a lower periodicity score as being more randomly generated.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter
    # A positive weight means people penalize periodicity (i.e. they prefer sequences with lower periodicity).
    beta_periodicity = pm.HalfNormal("beta_periodicity", sigma=5.0)

    # Calculate difference. If B is more periodic than A, diff > 0, increasing p_left.
    diff = periodicity_b - periodicity_a
    
    # Clip probability for numerical stability
    p = pm.math.sigmoid(beta_periodicity * diff)
    p_left_safe = pt.clip(p, 1e-6, 1 - 1e-6)
    p_left = pm.Deterministic("p_left", p_left_safe)

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
