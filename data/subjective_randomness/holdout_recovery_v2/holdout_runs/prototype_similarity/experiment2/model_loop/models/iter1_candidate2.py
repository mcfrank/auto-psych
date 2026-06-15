"""
People judge a sequence as more random when its transitions between outcomes are
closer to maximum uncertainty — when each outcome is equally likely to be
followed by the same or a different outcome. They choose whichever sequence has
higher Shannon entropy of the transition process (closer to 50% alternation
rate) as the more random-looking sequence.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Sensitivity to entropy differences
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Shannon entropy of the transition process for each sequence
    p_a = pt.clip(p_alts_a, 1e-6, 1 - 1e-6)
    p_b = pt.clip(p_alts_b, 1e-6, 1 - 1e-6)

    H_a = -p_a * pt.log(p_a) - (1 - p_a) * pt.log(1 - p_a)
    H_b = -p_b * pt.log(p_b) - (1 - p_b) * pt.log(1 - p_b)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (H_a - H_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
