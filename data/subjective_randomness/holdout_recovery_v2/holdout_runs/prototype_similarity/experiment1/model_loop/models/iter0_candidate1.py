"""
Maximum-run-length aversion hypothesis:
People judge a sequence as less random the longer its longest unbroken streak of
identical outcomes. When comparing two sequences, they choose the one with the
shorter normalized maximum run as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Sensitivity to run-length difference (inverse temperature).
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Shorter max run → more random → higher probability of being chosen.
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (max_run_norm_b - max_run_norm_a)),
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
