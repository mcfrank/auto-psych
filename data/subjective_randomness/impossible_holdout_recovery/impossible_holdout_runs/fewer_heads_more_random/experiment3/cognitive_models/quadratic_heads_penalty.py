"""People judge the randomness of a sequence strictly by the absolute number of heads it contains, but the penalty for heads grows quadratically, such that each additional head decreases perceived randomness more than the previous one."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Parameter for the sensitivity to the quadratic head count.
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Quadratic penalty
    score_a = -tau * pt.pow(pt.cast(h_a, "float64"), 2)
    score_b = -tau * pt.pow(pt.cast(h_b, "float64"), 2)

    p_left_raw = pm.math.sigmoid(score_a - score_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
