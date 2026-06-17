"""People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains, but their choices are subject to a constant lapse rate representing random guessing."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=5.0)
    lapse = pm.Beta("lapse", alpha=1.0, beta=9.0)

    score_a = pt.cast(h_a, "float64")
    score_b = pt.cast(h_b, "float64")

    p_logistic = pm.math.sigmoid(tau * (score_a - score_b))
    
    p_left = pm.Deterministic(
        "p_left", 
        lapse * 0.5 + (1.0 - lapse) * p_logistic
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
