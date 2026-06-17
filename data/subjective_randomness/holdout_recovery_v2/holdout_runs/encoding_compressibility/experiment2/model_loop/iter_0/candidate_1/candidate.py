"""
People judge a sequence as more random when it contains less periodic structure.
Sequences that repeat a regular cycle look non-random because the mind detects
the underlying period. When comparing two sequences, people choose the one with
lower periodicity as the more random-looking.
"""

import numpy as np
import pymc as pm

with pm.Model() as model:
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    tau = pm.HalfNormal("tau", sigma=2.0)

    # p_left is high when A has lower periodicity (looks more random) than B
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (periodicity_b - periodicity_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
