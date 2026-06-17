"""
People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. However, the perceived penalty for heads scales with the square root of the head count, meaning sensitivity diminishes gradually as the number of heads increases.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter for choice noise
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Calculate scores based on the square root of the number of heads.
    # Since they prefer fewer heads, a higher number of heads means a lower (more negative) score.
    # We cast to float64 to ensure pt.sqrt operates smoothly.
    score_a = -pt.sqrt(pt.cast(h_a, "float64"))
    score_b = -pt.sqrt(pt.cast(h_b, "float64"))

    # Choice probability using a logistic sigmoid
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
