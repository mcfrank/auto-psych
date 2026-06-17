"""People evaluate alternation deviation on the count scale rather than the
proportion scale — a sequence with two extra transitions relative to the ideal
count looks equally deviant regardless of whether it is 4 or 8 flips long, so
the randomness signal scales naturally with sequence length."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # Ideal alternation count for each sequence given its length
    ideal_a = theta_alt * pt.cast(n_a - 1, "float64")
    ideal_b = theta_alt * pt.cast(n_b - 1, "float64")

    # Quadratic deviation on the count scale (not proportion scale)
    score_a = -((pt.cast(alts_a, "float64") - ideal_a) ** 2)
    score_b = -((pt.cast(alts_b, "float64") - ideal_b) ** 2)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
