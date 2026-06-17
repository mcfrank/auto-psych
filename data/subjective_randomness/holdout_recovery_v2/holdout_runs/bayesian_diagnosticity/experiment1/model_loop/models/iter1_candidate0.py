"""
Gaussian prototype similarity on alternation rate: people judge a sequence as
more random-looking when its alternation rate is closer to an internal prototype
ideal, with proximity computed quadratically (Gaussian decay) rather than by
absolute deviation. Small departures from the prototype are disproportionately
forgiven compared to large ones, unlike the uniform per-unit penalty implied by
an absolute-value sensitivity function.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    score_a = -((p_alts_a - theta_alt) ** 2)
    score_b = -((p_alts_b - theta_alt) ** 2)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
