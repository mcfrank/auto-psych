"""Impossible ground-truth theory: longer runs => more random-looking.

The opposite of the gambler's-fallacy intuition (and of the compressibility
seed model, which *penalizes* long runs): this generator judges a sequence more
random the longer its longest streak of identical outcomes. NOT a plausible
model of human subjective-randomness judgment; a deliberately weird ground-truth
generator for the impossible-theory holdout analysis.

Same PyMC structure as the seed models; the score is the normalized maximum run
length alone (no free shape parameters), so the generating params are exactly
{beta, side_bias}.
"""

import numpy as np
import pymc as pm

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = max_run_norm_a
    score_b = max_run_norm_b

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
