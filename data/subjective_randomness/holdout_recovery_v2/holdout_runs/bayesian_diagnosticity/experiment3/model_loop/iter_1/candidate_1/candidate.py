"""
People judge a coin sequence as more random-looking based solely on the length
of its longest streak: longer maximum runs always feel less random, with no
ideal run length to aim for. When comparing two sequences, people simply choose
whichever has the shorter maximum run as the more random-looking one.
"""

import numpy as np
import pymc as pm

with pm.Model() as model:
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=2.0)

    # Prefer sequence with shorter max run; positive (max_run_b - max_run_a) → chose left
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (max_run_b - max_run_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
