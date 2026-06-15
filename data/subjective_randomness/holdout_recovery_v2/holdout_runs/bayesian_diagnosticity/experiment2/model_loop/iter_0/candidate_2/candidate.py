"""
People judge a sequence as more random-looking when its proportion of heads is
closer to 0.5. Balance — equal heads and tails — is the primary cue for
randomness; imbalanced sequences (too many heads or too many tails) look less
random. This single balance mechanism, not alternation or run structure, drives
the choice.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # imbalance = |p_heads - 0.5|; lower means more balanced (more random-looking)
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Sensitivity to balance differences
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Choose left when left is more balanced (imbalance_a < imbalance_b)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (imbalance_b - imbalance_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
