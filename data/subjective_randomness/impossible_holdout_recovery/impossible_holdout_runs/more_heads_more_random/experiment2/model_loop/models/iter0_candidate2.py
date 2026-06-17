"""
People evaluate the randomness of a sequence strictly based on the cubed number of heads it contains. This mechanism creates an extreme, accelerating non-linear preference where sequences with high head counts are overwhelmingly perceived as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=1.0)

    val_a = pt.cast(h_a, "float64") ** 3
    val_b = pt.cast(h_b, "float64") ** 3

    p_left_raw = pm.math.sigmoid(tau * (val_a - val_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)