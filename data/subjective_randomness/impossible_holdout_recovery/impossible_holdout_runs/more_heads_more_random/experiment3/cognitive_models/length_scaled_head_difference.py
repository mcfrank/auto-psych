"""People evaluate randomness primarily by the absolute number of heads, but their sensitivity to the difference in head counts between two sequences is diminished when the overall length of the sequences being compared is larger."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=5.0)

    # Sensitivity to difference is scaled by the total lengths
    total_n = pt.cast(n_a + n_b, "float64")
    total_n = pt.clip(total_n, 1e-3, 1e4) # numerical safety
    
    diff = pt.cast(h_a - h_b, "float64")

    p_left = pm.Deterministic(
        "p_left", 
        pm.math.sigmoid(tau * (diff / total_n))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
