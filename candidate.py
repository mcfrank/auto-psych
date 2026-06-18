"""
Observers judge a sequence as more random the closer its proportion of heads is to a subjective ideal proportion. Rather than expecting exactly a 50% split, they penalize sequences based on how far their proportion of heads deviates from an internal, potentially biased ideal rate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    ideal_p = pm.Beta("ideal_p", alpha=2.0, beta=2.0)
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Hypothesis: sequences closer to subjective ideal proportion are more random
    # We use negative absolute deviation as the subjective "randomness" score
    score_a = -pt.abs(p_a - ideal_p)
    score_b = -pt.abs(p_b - ideal_p)

    # Convert difference in scores to probability of choosing left
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
