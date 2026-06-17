"""
People judge a sequence as more random when its proportion of heads is closer
to 0.5 — when the sequence is more balanced. They ignore run structure,
alternation patterns, and periodicity entirely; only the overall H/T ratio
guides the judgment. The sequence with less imbalanced coin-flip counts looks
more random.
"""

import numpy as np
import pymc as pm

with pm.Model() as model:
    # Stimulus inputs — imbalance = |proportion_heads - 0.5| for each sequence
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Sensitivity: how strongly does imbalance drive the randomness judgment?
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Prefer the sequence with less imbalance (more balanced = more random)
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (imbalance_b - imbalance_a))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
