"""
People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. Their trial-by-trial comparison of these head counts follows a probit function, meaning their evaluation noise is normally rather than logistically distributed.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: absolute count of heads
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter representing sensitivity to head count differences
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Sequences with FEWER heads are judged as MORE random
    score_a = -tau * pt.cast(h_a, "float64")
    score_b = -tau * pt.cast(h_b, "float64")

    # Probit link function: cumulative normal distribution for decision noise
    p_left_raw = pm.math.invprobit(score_a - score_b)
    
    # Clip to avoid exact 0 or 1, ensuring numerical stability
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
