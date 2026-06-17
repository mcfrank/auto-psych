"""People judge the randomness of a sequence by its proportion of heads, holding a biased belief that sequences containing a lower proportion of heads are more representative of a random coin."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: proportion of heads
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter representing sensitivity to the head proportion difference.
    # Higher tau means stronger preference for sequences with lower proportion of heads.
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Sequences with LOWER proportion of heads are judged as MORE random.
    score_a = -tau * p_a
    score_b = -tau * p_b

    # Probability of choosing left (sequence A)
    p_left_raw = pm.math.sigmoid(score_a - score_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
