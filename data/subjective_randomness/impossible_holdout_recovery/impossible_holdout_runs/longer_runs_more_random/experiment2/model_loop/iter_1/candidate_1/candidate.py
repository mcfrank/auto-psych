"""People judge the randomness of a sequence simply by the absolute length of its longest streak of identical outcomes. They perceive a sequence as more random the longer its absolute maximum run is, completely ignoring the total length of the sequence in their evaluation."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=1.0)

    score_a = pt.cast(max_run_a, "float64")
    score_b = pt.cast(max_run_b, "float64")

    # Higher score = more random (perceive as more random the longer the run)
    p = pm.math.sigmoid(tau * (score_a - score_b))
    p_left = pm.Deterministic("p_left", pt.clip(p, 1e-6, 1 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
