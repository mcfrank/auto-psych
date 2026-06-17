"""People evaluate the randomness of a sequence based on the number of heads it contains, but their perception of randomness scales as an inferred power-law function of the head count."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=5.0)
    exponent = pm.HalfNormal("exponent", sigma=3.0)

    score_a = pt.cast(h_a, "float64") ** exponent
    score_b = pt.cast(h_b, "float64") ** exponent

    p_left = pm.Deterministic(
        "p_left", 
        pm.math.sigmoid(tau * (score_a - score_b))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
