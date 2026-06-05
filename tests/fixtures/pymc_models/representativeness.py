# file: representativeness.py
"""Observers prefer the sequence whose head-proportion is closer to 0.5
(simple representativeness heuristic), with a softmax decision rule."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=5.0)

    p_a = pt.cast(h_a, "float64") / pt.cast(n_a, "float64")
    p_b = pt.cast(h_b, "float64") / pt.cast(n_b, "float64")
    dev_a = pt.abs(p_a - 0.5)
    dev_b = pt.abs(p_b - 0.5)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dev_b - dev_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
