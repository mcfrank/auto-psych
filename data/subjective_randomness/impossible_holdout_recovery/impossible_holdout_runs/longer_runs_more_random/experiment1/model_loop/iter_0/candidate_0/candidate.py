"""
People judge a sequence as random based on its similarity to an ideal prototype, but they evaluate the sequence by its maximum absolute deviation (L-infinity norm) from ideal head balance and expected alternation rate, penalizing only its most salient flaw.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free parameters
    weight_h = pm.HalfNormal("weight_h", sigma=5.0)
    weight_alts = pm.HalfNormal("weight_alts", sigma=5.0)
    tau = pm.HalfNormal("tau", sigma=1.0)

    # L-infinity distance to prototype (0.5 for heads, 0.5 for alternations)
    dev_h_a = weight_h * pt.abs(p_a - 0.5)
    dev_alts_a = weight_alts * pt.abs(p_alts_a - 0.5)
    dist_a = pt.maximum(dev_h_a, dev_alts_a)

    dev_h_b = weight_h * pt.abs(p_b - 0.5)
    dev_alts_b = weight_alts * pt.abs(p_alts_b - 0.5)
    dist_b = pt.maximum(dev_h_b, dev_alts_b)

    # Probability of choosing left (smaller distance is more random)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dist_b - dist_a)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
