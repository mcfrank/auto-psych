"""
Periodicity aversion: people choose the sequence with lower periodic structure as
more random. The single free parameter tau captures sensitivity to periodicity
differences between the two sequences.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Sensitivity to periodicity differences (positive: less periodic → more random).
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Lower periodicity in A relative to B → A looks more random → chose_left=1.
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (periodicity_b - periodicity_a)),
    )

    # Observed response.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
