"""
People judge the randomness of a sequence solely by evaluating the overall balance of its outcomes. They perceive a sequence as more random the closer its proportion of heads is to exactly fifty percent, strictly penalizing any deviation from perfect balance.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs - absolute deviation from equal head/tail balance
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter mapping subjective difference to choice probability
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Sequences with lower imbalance are perceived as more random
    score_a = -tau * imbalance_a
    score_b = -tau * imbalance_b

    # Probability of choosing the left sequence (A)
    p_left_raw = pm.math.sigmoid(score_a - score_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
