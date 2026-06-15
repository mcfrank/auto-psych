"""
Alternation-rate-only prototype similarity: people judge a sequence's
randomness based solely on how close its alternation rate is to an internal
prototype ideal. H/T balance provides no additional information once
alternation rate is accounted for.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = -pt.abs(p_alts_a - theta_alt)
    score_b = -pt.abs(p_alts_b - theta_alt)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
