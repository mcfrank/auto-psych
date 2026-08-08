"""PyMC adapter for the Hahn & Warren (2009) finite-experience family.

Randomness = log expected probability that the sequence occurs at least once
within a finite global stream of fair flips, with the experience length
uncertain over {10, 20, 50} via stick-breaking weights (w_10 = short_weight;
w_20 = (1 - short_weight) * mid_share; w_50 = remainder). The occ_n* columns
are exact occurrence probabilities precomputed by the featurizer. See the
pure-Python twin in ``model_families/finite_experience_occurrence.py`` for
the full rationale and the cross-length caveat.
"""

import numpy as np
import pymc as pm

with pm.Model() as model:
    occ_n10_a = pm.Data("occ_n10_a", np.zeros(1, dtype="float64"))
    occ_n20_a = pm.Data("occ_n20_a", np.zeros(1, dtype="float64"))
    occ_n50_a = pm.Data("occ_n50_a", np.zeros(1, dtype="float64"))
    occ_n10_b = pm.Data("occ_n10_b", np.zeros(1, dtype="float64"))
    occ_n20_b = pm.Data("occ_n20_b", np.zeros(1, dtype="float64"))
    occ_n50_b = pm.Data("occ_n50_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    short_weight = pm.Uniform("short_weight", lower=0.01, upper=0.99)
    mid_share = pm.Uniform("mid_share", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    w10 = short_weight
    w20 = (1.0 - short_weight) * mid_share
    w50 = (1.0 - short_weight) * (1.0 - mid_share)

    score_a = pm.math.log(w10 * occ_n10_a + w20 * occ_n20_a + w50 * occ_n50_a)
    score_b = pm.math.log(w10 * occ_n10_b + w20 * occ_n20_b + w50 * occ_n50_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
