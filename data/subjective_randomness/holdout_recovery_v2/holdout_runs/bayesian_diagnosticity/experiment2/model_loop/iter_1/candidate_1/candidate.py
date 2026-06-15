"""
People judge a sequence as more random-looking when it contains less periodic
structure. A truly random sequence should have no detectable rhythm or cycle,
so a sequence with high periodicity looks engineered rather than chance-produced.
When comparing two sequences, people choose the one with lower periodicity as
the more random-looking one.
"""

import numpy as np
import pymc as pm

with pm.Model() as model:
    periodicity_a = pm.Data("periodicity_a", np.zeros(1))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1))

    tau = pm.HalfNormal("tau", sigma=2.0)

    # Higher periodicity_b relative to periodicity_a means A looks more random,
    # so probability of choosing left (A) increases.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (periodicity_b - periodicity_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
