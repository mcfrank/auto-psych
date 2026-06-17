"""
People evaluate the randomness of a sequence based on the absolute number of heads it contains, but following the Weber-Fechner law for numerosity perception, their sensitivity to additional heads exhibits diminishing returns. When comparing two sequences, they compute a randomness score that grows logarithmically with the absolute head count plus a constant.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter
    tau = pm.HalfNormal("tau", sigma=1.0)
    
    # Logarithmic transformation of head counts
    log_h_a = pt.log(h_a + 1.0)
    log_h_b = pt.log(h_b + 1.0)

    # Choice probability based on the difference in logarithmic scores
    p_left_raw = pm.math.sigmoid(tau * (log_h_a - log_h_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
