"""People judge the randomness of a sequence by comparing its feature proportions to a subjective ideal using Bayesian smoothing, but their sensitivity to these deviations follows a Gaussian-like generalization gradient. Instead of penalizing deviations linearly, they disproportionately penalize sequences that exhibit extreme, glaring deviations on any single feature dimension, while easily tolerating small differences from the prototype."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Feature inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))

    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Cognitive parameters
    # The ideal alternation rate people expect
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    # The relative weight of alternations vs. heads balance
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    balance_weight = 1.0 - alt_weight

    # Pseudo-counts for Bayesian smoothing of the proportions
    # Higher K means more regularization toward the ideal (stronger prior)
    K_bal = pm.Uniform("K_bal", lower=0.0, upper=20.0)
    K_alt = pm.Uniform("K_alt", lower=0.0, upper=20.0)

    # Scaling parameter and side bias for the decision rule
    beta = pm.Uniform("beta", lower=0.1, upper=50.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Calculate denominators
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")

    n_alts_a_f = pt.cast(pt.maximum(n_a - 1, 1), "float64")
    n_alts_b_f = pt.cast(pt.maximum(n_b - 1, 1), "float64")

    # Smoothed deviations: (n / (n + K)) * raw_deviation
    # For heads/tails balance, raw deviation is imbalance = |h/n - 0.5|
    smooth_imb_a = (n_a_f / (n_a_f + K_bal)) * imbalance_a
    smooth_imb_b = (n_b_f / (n_b_f + K_bal)) * imbalance_b

    # For alternations, raw deviation is |p_alts - theta_alt|
    raw_alt_dev_a = pt.abs(p_alts_a - theta_alt)
    raw_alt_dev_b = pt.abs(p_alts_b - theta_alt)

    smooth_alt_dev_a = (n_alts_a_f / (n_alts_a_f + K_alt)) * raw_alt_dev_a
    smooth_alt_dev_b = (n_alts_b_f / (n_alts_b_f + K_alt)) * raw_alt_dev_b

    # Score sequences based on their SQUARED smoothed distance from the ideal 
    # (higher distance = lower score)
    score_a = -(balance_weight * (smooth_imb_a ** 2) + alt_weight * (smooth_alt_dev_a ** 2))
    score_b = -(balance_weight * (smooth_imb_b ** 2) + alt_weight * (smooth_alt_dev_b ** 2))

    # Softmax decision rule
    p_left_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
