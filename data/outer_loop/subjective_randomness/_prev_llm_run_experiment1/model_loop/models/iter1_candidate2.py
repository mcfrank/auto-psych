"""
People judge a sequence as more random when it has lower periodicity — fewer
regular, repeating cycles in the pattern of outcomes. A sequence that follows
a repeating temporal template feels obviously non-random; the sequence with
less periodic structure looks more like genuine noise, regardless of its
overall balance or run lengths.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Sensitivity to differences in periodicity.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Lower periodicity → more random: chose_left when A is less periodic than B.
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (periodicity_b - periodicity_a))
    )

    # Observed response.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
