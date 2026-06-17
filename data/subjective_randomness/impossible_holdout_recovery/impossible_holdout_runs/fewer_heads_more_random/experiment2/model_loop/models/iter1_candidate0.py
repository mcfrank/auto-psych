"""
People judge the randomness of a sequence strictly by the absolute number of heads it contains, but their sensitivity to this count diminishes logarithmically. Sequences are penalized based on the logarithm of their head count, meaning the perceived difference in randomness between 1 and 2 heads is larger than the difference between 8 and 9.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter: sensitivity to the difference in logarithmic head count
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Sequences are penalized based on the logarithm of the number of heads.
    # We add 1.0 to avoid log(0) since head counts can be 0.
    score_a = -pt.log(pt.cast(h_a, 'float64') + 1.0)
    score_b = -pt.log(pt.cast(h_b, 'float64') + 1.0)

    # Probability of choosing sequence A (left)
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    
    # Clip for numerical stability
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
