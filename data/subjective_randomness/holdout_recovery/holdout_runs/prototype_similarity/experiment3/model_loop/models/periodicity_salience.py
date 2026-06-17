"""Periodicity-salience model: randomness judged via template-matching.

All competitive models from experiment 2 rely on alternation statistics
(p_alts deviation or runs z-score) and imbalance. This model uses a
qualitatively different cue: how closely the sequence matches a short
repeating template, captured by the precomputed periodicity feature. A highly
periodic sequence (e.g., HTHTTHTH matching HTHTHTHT) feels non-random because
it exhibits discernible structure rather than independent outcomes. Periodicity
and alternation rate can dissociate: a sequence can be highly alternating
without matching any template, or match a period-2 template while being
moderately alternating. Comparing this model against alternation-based models
tests whether template-detection is an independent component of randomness
judgments.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    periodicity_weight = pm.Uniform("periodicity_weight", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - periodicity_weight

    score_a = -(periodicity_weight * periodicity_a + balance_weight * imbalance_a)
    score_b = -(periodicity_weight * periodicity_b + balance_weight * imbalance_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
