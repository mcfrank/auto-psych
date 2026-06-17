"""
People judge a sequence as more random when its alternation rate is closest to the
prototype value of 0.5 — the expected alternation rate for a fair coin. Randomness
perception follows a Gaussian similarity function: the closer a sequence's p_alts
is to 0.5, the more random it looks, with similarity falling off symmetrically as a
function of squared deviation from the prototype. On each trial, people choose
whichever of the two sequences is nearer to this internal prototype.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Precision of the Gaussian similarity function around the prototype.
    tau = pm.HalfNormal("tau", sigma=5.0)

    dev_a = (p_alts_a - 0.5) ** 2
    dev_b = (p_alts_b - 0.5) ** 2

    # p_left is higher when sequence A is closer to the prototype than B.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dev_b - dev_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
