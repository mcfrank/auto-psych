"""
People judge the randomness of a sequence by evaluating its distance from an ideal subjective prototype, but their sensitivity to deviations follows a psychophysical power law with a compressive exponent. This sub-linear scaling means that while initial departures from the prototype are penalized, further deviations toward extremes (like perfect imbalance) result in diminishing additional penalties.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — using the same features as prototype_similarity
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Compressive exponent (constrained to be sub-linear)
    alpha = pm.Uniform("alpha", lower=0.1, upper=1.0)

    balance_weight = 1.0 - alt_weight

    # Linear distance from prototype
    dist_a = balance_weight * imbalance_a + alt_weight * pt.abs(p_alts_a - theta_alt)
    dist_b = balance_weight * imbalance_b + alt_weight * pt.abs(p_alts_b - theta_alt)

    # Sub-linear penalty via psychophysical power law
    eps = 1e-6
    score_a = -pt.pow(dist_a + eps, alpha)
    score_b = -pt.pow(dist_b + eps, alpha)

    # Choice probability
    p_left_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
