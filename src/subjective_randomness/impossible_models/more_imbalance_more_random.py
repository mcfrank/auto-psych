"""Impossible ground-truth theory: more imbalance => more random-looking.

The opposite of the representativeness heuristic: this generator judges a
sequence more random the more lopsided its heads/tails split (further from
50/50). NOT a plausible model of human subjective-randomness judgment; a
deliberately weird ground-truth generator for the impossible-theory holdout
analysis.

Same PyMC structure as the seed models; the score is the head/tail imbalance
alone (no free shape parameters), so the generating params are exactly
{beta, side_bias}.
"""

import numpy as np
import pymc as pm

with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = imbalance_a
    score_b = imbalance_b

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
