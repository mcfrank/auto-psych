"""Impossible ground-truth theory: more heads => more random-looking.

NOT a plausible model of how humans judge subjective randomness — a deliberately
weird ground-truth generator for the impossible-theory holdout analysis. The
loop's models (alternation / balance / compressibility / diagnosticity) should
never recover it, so the held-out correlation should stay low.

Same PyMC structure as the seed models (feature pm.Data containers, free
beta/side_bias, deterministic score, sigmoid p_left, Bernoulli response). The
score is the head count alone — no free shape parameters — so the generating
params are exactly {beta, side_bias}. Mirrors the head-count process in
projects/subjective_randomness/ground_truth_models.py::prefer_more_heads, ported
to the PyMC fixed-param generator interface.
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

    score_a = pt.cast(h_a, "float64")
    score_b = pt.cast(h_b, "float64")

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
