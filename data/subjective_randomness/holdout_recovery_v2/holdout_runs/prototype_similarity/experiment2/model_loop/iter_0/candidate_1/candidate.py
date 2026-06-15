"""
People judge a sequence as more random when its proportion of heads is closer to 0.5.
When comparing two sequences, they choose the one whose head count deviates less from
an equal split as more random — imbalance is the sole cue driving the choice.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — imbalance of each sequence (|h/n - 0.5|, higher = more imbalanced).
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Sensitivity: how strongly imbalance difference drives the choice.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Chose left (A) when A is less imbalanced than B.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (imbalance_b - imbalance_a)))

    # Observed response.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
