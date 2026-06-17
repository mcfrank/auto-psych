"""
People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. Their trial-by-trial comparison of these head counts follows a Cauchy CDF, meaning their evaluation noise is heavy-tailed and they are prone to occasional extreme deviations from their core preference.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Calculate difference in heads.
    # Positive difference means sequence B has more heads than sequence A.
    # Because people prefer fewer heads, a positive difference increases the probability of choosing A (left).
    diff = pt.cast(h_b - h_a, "float64")
    
    # Cauchy CDF: F(x) = 0.5 + (1 / pi) * arctan(x)
    p_left_raw = 0.5 + (1.0 / np.pi) * pt.arctan(tau * diff)
    
    # Clamp probability to be strictly between 0 and 1 for numerical safety
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
