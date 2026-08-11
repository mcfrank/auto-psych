"""PyMC adapter for the Kahneman & Tversky local-representativeness family.

This quantitative operationalization separates K&T's local-balance and
irregularity claims. Balance is averaged over the whole sequence and sliding
windows at scales two through four. Irregularity combines distance from an
over-alternating prototype with an explicit periodic-template penalty. See the
pure-Python twin in
``model_families/local_representativeness.py`` for the full rationale.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    multiscale_imbalance_a = pm.Data(
        "multiscale_imbalance_a", np.zeros(1, dtype="float64")
    )
    multiscale_imbalance_b = pm.Data(
        "multiscale_imbalance_b", np.zeros(1, dtype="float64")
    )
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    theta_alt = pm.Uniform("theta_alt", lower=0.5001, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    periodic_share = pm.Uniform("periodic_share", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight
    irregularity_a = (
        (1.0 - periodic_share) * pt.abs(p_alts_a - theta_alt)
        + periodic_share * periodicity_a
    )
    irregularity_b = (
        (1.0 - periodic_share) * pt.abs(p_alts_b - theta_alt)
        + periodic_share * periodicity_b
    )
    score_a = -(
        balance_weight * multiscale_imbalance_a + alt_weight * irregularity_a
    )
    score_b = -(
        balance_weight * multiscale_imbalance_b + alt_weight * irregularity_b
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
