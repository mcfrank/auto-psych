"""
People judge a sequence as more random-looking when it contains less detectable
periodic structure. A truly random coin sequence should not repeat a regular
pattern, so sequences with stronger periodicity feel non-random. When comparing
two sequences, people choose the one with lower periodicity as the more
random-looking one.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Free parameter: sensitivity to periodicity differences.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Higher periodicity in B → A looks more random → p_left rises.
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (periodicity_b - periodicity_a))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
