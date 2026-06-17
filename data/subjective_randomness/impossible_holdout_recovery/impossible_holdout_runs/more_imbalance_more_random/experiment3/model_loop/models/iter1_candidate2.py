"""
People judge sequence randomness based purely on the Shannon entropy of the sequence's outcome frequencies, paradoxically perceiving sequences with lower entropy (a more biased distribution of heads and tails) as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    tau = pm.HalfNormal("tau", sigma=5.0)

    p_a_safe = pt.clip(p_a, 1e-6, 1.0 - 1e-6)
    p_b_safe = pt.clip(p_b, 1e-6, 1.0 - 1e-6)
    
    entropy_a = -p_a_safe * pt.log(p_a_safe) - (1.0 - p_a_safe) * pt.log(1.0 - p_a_safe)
    entropy_b = -p_b_safe * pt.log(p_b_safe) - (1.0 - p_b_safe) * pt.log(1.0 - p_b_safe)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (entropy_b - entropy_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
