"""
People judge the randomness of a sequence strictly by the absolute number of heads it contains, rather than the proportion of heads. Sequences are penalized directly for their total count of heads, such that fewer absolute heads are judged as more representative of a random process, regardless of the sequence's length.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter: sensitivity to the number of heads
    # (inference fits it).
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Sequences are penalized based on their absolute number of heads.
    # Fewer heads means a higher (less negative) score.
    score_a = -pt.cast(h_a, 'float64')
    score_b = -pt.cast(h_b, 'float64')

    # Probability of choosing sequence A (left)
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    
    # Clip for numerical stability
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response: the pm.Data tensor is passed directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
