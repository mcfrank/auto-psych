"""PyMC model for the asymmetric prototype-similarity hypothesis.

Random-looking sequences are judged by their similarity to a prototype with balanced counts and an ideal alternation rate, but the penalty for deviating from the ideal alternation rate is asymmetric: sequences that alternate less than the ideal are penalized more harshly than those that alternate more.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    # Asymmetry parameter: < 0.5 means over-alternation is penalized less than under-alternation
    asym = pm.Uniform("asym", lower=0.01, upper=0.99)

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight

    # Asymmetric distance for alternation rate
    diff_a = p_alts_a - theta_alt
    alt_penalty_a = pm.math.switch(
        diff_a > 0, 2.0 * asym * diff_a, -2.0 * (1.0 - asym) * diff_a
    )

    diff_b = p_alts_b - theta_alt
    alt_penalty_b = pm.math.switch(
        diff_b > 0, 2.0 * asym * diff_b, -2.0 * (1.0 - asym) * diff_b
    )

    score_a = -(balance_weight * imbalance_a + alt_weight * alt_penalty_a)
    score_b = -(balance_weight * imbalance_b + alt_weight * alt_penalty_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )

    # Likelihood
    pm.Bernoulli("response", p=p_left, observed=chose_left)
