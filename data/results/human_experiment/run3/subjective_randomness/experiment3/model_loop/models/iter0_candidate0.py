"""People judge the randomness of a sequence by comparing its smoothed feature proportions to a subjective ideal, but instead of expecting exact perfection, they expect a 'typical' amount of sampling variation. They penalize sequences based on the absolute difference between their smoothed deviations and these expected typical deviations, naturally penalizing exactly balanced sequences as suspiciously artificial."""

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

    # The expected typical deviation for imbalance and alternations
    expected_imbalance = pm.Uniform("expected_imbalance", lower=0.0, upper=0.4)
    expected_alt_dev = pm.Uniform("expected_alt_dev", lower=0.0, upper=0.4)

    # The relative weight of alternations vs. heads balance
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    balance_weight = 1.0 - alt_weight

    # Pseudo-counts for Bayesian smoothing of the proportions
    K_bal = pm.Uniform("K_bal", lower=0.0, upper=20.0)
    K_alt = pm.Uniform("K_alt", lower=0.0, upper=20.0)

    # Scaling parameter and side bias for the decision rule
    beta = pm.Uniform("beta", lower=0.1, upper=20.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Calculate denominators
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")

    n_alts_a_f = pt.cast(pt.maximum(n_a - 1, 1), "float64")
    n_alts_b_f = pt.cast(pt.maximum(n_b - 1, 1), "float64")

    # Smoothed deviations: (n / (n + K)) * raw_deviation
    smooth_imb_a = (n_a_f / (n_a_f + K_bal)) * imbalance_a
    smooth_imb_b = (n_b_f / (n_b_f + K_bal)) * imbalance_b

    raw_alt_dev_a = pt.abs(p_alts_a - theta_alt)
    raw_alt_dev_b = pt.abs(p_alts_b - theta_alt)

    smooth_alt_dev_a = (n_alts_a_f / (n_alts_a_f + K_alt)) * raw_alt_dev_a
    smooth_alt_dev_b = (n_alts_b_f / (n_alts_b_f + K_alt)) * raw_alt_dev_b

    # Penalize based on distance from expected typical deviation
    dist_imb_a = pt.abs(smooth_imb_a - expected_imbalance)
    dist_imb_b = pt.abs(smooth_imb_b - expected_imbalance)

    dist_alt_a = pt.abs(smooth_alt_dev_a - expected_alt_dev)
    dist_alt_b = pt.abs(smooth_alt_dev_b - expected_alt_dev)

    # Score sequences (higher distance = lower score/randomness)
    score_a = -(balance_weight * dist_imb_a + alt_weight * dist_alt_a)
    score_b = -(balance_weight * dist_imb_b + dist_alt_b * alt_weight)

    # Softmax decision rule
    p_left_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
