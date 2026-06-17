"""
People evaluate the randomness of a sequence based on the square root of the number of heads it contains, reflecting a diminishing sensitivity to each additional head, and their choices are subject to a constant lapse rate representing occasional random guessing.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=1.0)
    lapse = pm.Beta("lapse", alpha=1.0, beta=9.0)

    # Hypothesis mechanism: square root of heads
    # Use pt.cast to float64 to ensure smooth gradients
    val_a = pt.sqrt(pt.cast(h_a, "float64"))
    val_b = pt.sqrt(pt.cast(h_b, "float64"))

    # Choice probability with lapse rate
    p_left_base = pm.math.sigmoid(tau * (val_a - val_b))
    p_left = pm.Deterministic(
        "p_left", 
        pt.clip((lapse / 2.0) + (1.0 - lapse) * p_left_base, 1e-6, 1.0 - 1e-6)
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
