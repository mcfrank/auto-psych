"""
People judge a sequence as more random when its proportion of heads is closer to 50%.
When comparing two sequences, they choose whichever has better head/tail balance —
the smaller absolute deviation from equal proportions — as the more random one.
A single inverse-temperature parameter governs how sensitively this balance
difference drives the choice.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    imbalance_a = pm.Data("imbalance_a", np.zeros(1))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1))

    # Sensitivity to head/tail balance difference.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Sequence A is preferred (left) when it has lower imbalance (closer to 50/50).
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (imbalance_b - imbalance_a))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
