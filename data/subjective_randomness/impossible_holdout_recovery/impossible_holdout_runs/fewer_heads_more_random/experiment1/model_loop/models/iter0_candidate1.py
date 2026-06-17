"""
People judge the randomness of a sequence by looking solely at the length of its longest streak of identical outcomes. Relying on a representativeness heuristic, they expect short runs and penalize sequences with longer streaks, judging sequences with a shorter maximum run length as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: maximum run length for sequence A and B
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter for sensitivity to streak length differences
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Difference in maximum run lengths
    # A positive difference means sequence B has a longer maximum run than sequence A.
    # Because shorter runs are judged as more random, sequence A should be chosen more often,
    # making p_left > 0.5.
    diff = pt.cast(max_run_b - max_run_a, "float64")
    p_left_raw = pm.math.sigmoid(tau * diff)
    
    # Clip for numerical safety
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
