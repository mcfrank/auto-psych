"""
People judge a sequence as more random when it contains less detectable
periodic structure. When comparing two sequences, they choose the one with
lower periodicity as the more random-looking one — periodic regularity
signals a non-random, patterned generator.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    periodicity_a = pm.Data("periodicity_a", np.zeros(1))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1))

    # Sensitivity to periodicity difference
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Lower periodicity → more random → more likely chosen
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (periodicity_b - periodicity_a)),
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
