"""
People judge the randomness of a sequence based purely on its absolute count of heads, rather than the proportion of heads. They hold a biased belief that sequences containing fewer total heads are more representative of a random coin, and thus penalize sequences based directly on their total head count independent of sequence length.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: absolute count of heads
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter representing the sensitivity to the head count difference.
    # Higher tau means stronger preference for sequences with fewer heads.
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Sequences with FEWER heads are judged as MORE random.
    # Therefore, we assign a higher score to lower absolute counts of heads.
    score_a = -tau * h_a
    score_b = -tau * h_b

    # Probability of choosing left (sequence A)
    p_left_raw = pm.math.sigmoid(score_a - score_b)
    # Clip to avoid exact 0 or 1, ensuring numerical stability
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
