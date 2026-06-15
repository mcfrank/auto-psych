"""People judge a sequence as more random when its alternation rate is closer (L1 distance)
to an internalized prototype value, where that prototype is learned and may be biased above
0.5 due to the well-documented human tendency to overestimate alternation in random sequences."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Where in alternation-rate space does "random" live for this participant?
    # Uniform over (0.35, 0.95) to allow for the alternation bias without excluding 0.5.
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    beta = pm.HalfNormal("beta", sigma=5.0)

    # L1 prototype similarity: smaller distance to prototype = more random
    score_a = -pt.abs(p_alts_a - theta_alt)
    score_b = -pt.abs(p_alts_b - theta_alt)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b)),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
