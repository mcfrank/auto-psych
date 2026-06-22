"""Random-looking sequences are judged by their deviation from a messy prototype (an ideal positive imbalance and ideal alternation rate), but the cognitive penalty grows as a quartic (power of 4) function of the deviation rather than a quadratic one. This different functional form creates a much wider, flatter tolerance for near-ideal sequences but imposes significantly harsher penalties on extreme deviations like exact balance or severe imbalance."""

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
    theta_imb = pm.Uniform("theta_imb", lower=0.0, upper=1.0)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    # Asymmetry parameter: < 0.5 means over-alternation is penalized less than under-alternation
    asym = pm.Uniform("asym", lower=0.01, upper=0.99)

    beta = pm.Uniform("beta", lower=0.2, upper=1000.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight

    # Quartic asymmetric distance for alternation rate
    diff_alt_a = p_alts_a - theta_alt
    alt_penalty_a = pm.math.switch(
        diff_alt_a > 0,
        2.0 * asym * (diff_alt_a**4),
        2.0 * (1.0 - asym) * (diff_alt_a**4),
    )

    diff_alt_b = p_alts_b - theta_alt
    alt_penalty_b = pm.math.switch(
        diff_alt_b > 0,
        2.0 * asym * (diff_alt_b**4),
        2.0 * (1.0 - asym) * (diff_alt_b**4),
    )

    # Quartic distance for messy imbalance prototype
    imb_penalty_a = (imbalance_a - theta_imb) ** 4
    imb_penalty_b = (imbalance_b - theta_imb) ** 4

    score_a = -(balance_weight * imb_penalty_a + alt_weight * alt_penalty_a)
    score_b = -(balance_weight * imb_penalty_b + alt_weight * alt_penalty_b)

    # Sigmoid link to probability with safety clip
    p_left_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Likelihood
    pm.Bernoulli("response", p=p_left, observed=chose_left)
