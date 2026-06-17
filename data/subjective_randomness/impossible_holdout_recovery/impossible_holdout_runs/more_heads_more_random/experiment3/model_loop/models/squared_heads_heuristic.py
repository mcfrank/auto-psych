"""People evaluate the randomness of a sequence strictly based on the squared number of heads it contains, amplifying the perception of randomness for sequences with very high head counts."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=1.0)

    score_a = pt.cast(h_a ** 2, "float64")
    score_b = pt.cast(h_b ** 2, "float64")

    p_left = pm.Deterministic(
        "p_left", 
        pm.math.sigmoid(tau * (score_a - score_b))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
