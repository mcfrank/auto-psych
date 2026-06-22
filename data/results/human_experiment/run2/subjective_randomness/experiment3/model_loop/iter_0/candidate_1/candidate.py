"""
Random-looking sequences are judged by an evidence accumulation process where each item in the sequence provides a baseline weight of evidence for randomness. This accumulated evidence is then discounted by the sequence's deviation from a messy prototype, which is characterized by an ideal positive imbalance and an aversion to long streaks of identical outcomes.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    w_base = pm.HalfNormal("w_base", sigma=1.0)
    w_imb = pm.HalfNormal("w_imb", sigma=5.0)
    ideal_imb = pm.HalfNormal("ideal_imb", sigma=0.5)
    w_max_run = pm.HalfNormal("w_max_run", sigma=5.0)

    # Compute value for each sequence
    v_a = (
        n_a * w_base
        - w_imb * (imbalance_a - ideal_imb) ** 2
        - w_max_run * max_run_norm_a
    )
    v_b = (
        n_b * w_base
        - w_imb * (imbalance_b - ideal_imb) ** 2
        - w_max_run * max_run_norm_b
    )

    # Choice probability
    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(v_a - v_b), 1e-6, 1 - 1e-6)
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
