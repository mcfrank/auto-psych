"""
Head-balance hypothesis: people choose the sequence whose proportion of heads
is closer to 0.5 as more random. Only head-count imbalance drives the choice;
alternation rate, run lengths, and all other sequence structure are ignored.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Imbalance: absolute deviation of head proportion from 0.5, for each sequence.
    # Lower imbalance → more balanced → looks more random.
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Sensitivity: how strongly participants respond to the imbalance difference.
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Sequence A appears more random when its imbalance is lower than B's.
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (imbalance_b - imbalance_a)),
    )

    # Observed choice: 1 = chose left (sequence A), 0 = chose right (sequence B).
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
