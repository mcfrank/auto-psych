"""People judge the randomness of a sequence by looking solely at the length of its longest streak of identical outcomes, judging sequences with a shorter maximum run length as more random."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: max run length
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter representing the sensitivity to max run length
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Sequences with shorter maximum runs are judged as MORE random.
    score_a = -tau * max_run_a
    score_b = -tau * max_run_b

    # Probability of choosing left (sequence A)
    p_left_raw = pm.math.sigmoid(score_a - score_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
