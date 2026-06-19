"""People judge a sequence as more random the higher its proportion of alternations, evaluating randomness via a linear monotonic preference rather than calculating distance to a subjective ideal alternation rate."""

import numpy as np
import pymc as pm

with pm.Model() as model:
    # Stimulus inputs for the proportion of alternations
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter representing the strength of the alternation preference.
    # HalfNormal explicitly enforces the monotonic preference for MORE alternations.
    tau = pm.HalfNormal("tau", sigma=10.0)

    # The utility is simply the proportion of alternations.
    # Higher proportion of alternations -> higher utility -> perceived as more random.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (p_alts_a - p_alts_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
