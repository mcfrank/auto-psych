"""Impossible ground-truth theory: fewer heads => more random-looking.

The sign-flipped mirror of more_heads_more_random — useful as a control with the
opposite head-count direction. NOT a plausible model of human subjective-
randomness judgment; a deliberately weird ground-truth generator for the
impossible-theory holdout analysis.

Same PyMC structure as the seed models; the score is the negated head count
alone (no free shape parameters), so the generating params are exactly
{beta, side_bias}.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = -pt.cast(h_a, "float64")
    score_b = -pt.cast(h_b, "float64")

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
