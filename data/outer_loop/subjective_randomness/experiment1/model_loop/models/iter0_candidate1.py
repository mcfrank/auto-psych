"""
People judge which sequence looks more random by focusing on the longest
unbroken run of identical outcomes. The sequence with the shorter maximum run
length appears more random, because long streaks are the most perceptually
salient violation of randomness expectations. All other structural features
of the sequence are ignored.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — normalized maximum run length for each sequence.
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Sensitivity to run-length differences (positive = shorter run → more random).
    beta = pm.HalfNormal("beta", sigma=5.0)

    # Sequence A looks more random when its max run is shorter than B's.
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (max_run_norm_b - max_run_norm_a)),
    )

    # Observed response.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
