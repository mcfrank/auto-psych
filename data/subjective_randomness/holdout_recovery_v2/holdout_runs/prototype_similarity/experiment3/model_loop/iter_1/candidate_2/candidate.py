"""
People judge a sequence as more random when its longest unbroken streak of the
same outcome is shorter. They compare the two sequences' maximum run lengths
directly and choose the one with the shorter streak, with a single sensitivity
parameter governing how sharply they discriminate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    # Sensitivity to max-run differences.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Sequence A (left) looks more random when its max run is shorter.
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (max_run_b - max_run_a)),
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
