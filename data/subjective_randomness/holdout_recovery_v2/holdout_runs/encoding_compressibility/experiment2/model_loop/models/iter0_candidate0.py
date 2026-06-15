"""
People judge a sequence as more random when it is close to a prototype with balanced
H/T counts and an ideal alternation rate, where closeness is measured by squared
deviation rather than absolute deviation. The randomness gradient is steepest at the
prototype — modest departures are penalized disproportionately — rather than declining
at a constant rate regardless of distance from ideal.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    # Beta needs a wider range than the L1 model because squared distances are
    # smaller in magnitude (distance^2 < distance for distances in [0, 1]).
    beta = pm.Uniform("beta", lower=0.5, upper=30.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight
    score_a = -(
        balance_weight * imbalance_a ** 2 + alt_weight * (p_alts_a - theta_alt) ** 2
    )
    score_b = -(
        balance_weight * imbalance_b ** 2 + alt_weight * (p_alts_b - theta_alt) ** 2
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
