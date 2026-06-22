"""People judge the randomness of a sequence by its smoothed distance from a subjective prototype, but their ideal template includes an expectation for higher-order alternating motifs (like HTHT) in addition to basic heads/tails balance and simple bigram alternations."""

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

    alt_motifs_a = pm.Data("alt_motifs_a", np.zeros(1, dtype="int64"))
    alt_motifs_b = pm.Data("alt_motifs_b", np.zeros(1, dtype="int64"))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Cognitive parameters
    # The ideal alternation rate and ideal alternating motif rate people expect
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    theta_alt_motif = pm.Uniform("theta_alt_motif", lower=0.0, upper=0.95)

    # Relative weights of balance, alternations, and alternating motifs
    weights = pm.Dirichlet("weights", a=np.ones(3))
    balance_weight = weights[0]
    alt_weight = weights[1]
    motif_weight = weights[2]

    # Pseudo-counts for Bayesian smoothing of the proportions
    K_bal = pm.Uniform("K_bal", lower=0.0, upper=20.0)
    K_alt = pm.Uniform("K_alt", lower=0.0, upper=20.0)
    K_motif = pm.Uniform("K_motif", lower=0.0, upper=20.0)

    # Scaling parameter and side bias for the decision rule
    beta = pm.Uniform("beta", lower=0.1, upper=20.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Calculate denominators
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")

    n_alts_a_f = pt.cast(pt.maximum(n_a - 1, 1), "float64")
    n_alts_b_f = pt.cast(pt.maximum(n_b - 1, 1), "float64")

    n_alt_motifs_a_f = pt.cast(pt.maximum(n_a - 3, 1), "float64")
    n_alt_motifs_b_f = pt.cast(pt.maximum(n_b - 3, 1), "float64")

    p_alt_motifs_a = pt.cast(alt_motifs_a, "float64") / n_alt_motifs_a_f
    p_alt_motifs_b = pt.cast(alt_motifs_b, "float64") / n_alt_motifs_b_f

    # Smoothed deviations: (n / (n + K)) * raw_deviation
    smooth_imb_a = (n_a_f / (n_a_f + K_bal)) * imbalance_a
    smooth_imb_b = (n_b_f / (n_b_f + K_bal)) * imbalance_b

    raw_alt_dev_a = pt.abs(p_alts_a - theta_alt)
    raw_alt_dev_b = pt.abs(p_alts_b - theta_alt)
    smooth_alt_dev_a = (n_alts_a_f / (n_alts_a_f + K_alt)) * raw_alt_dev_a
    smooth_alt_dev_b = (n_alts_b_f / (n_alts_b_f + K_alt)) * raw_alt_dev_b

    raw_motif_dev_a = pt.abs(p_alt_motifs_a - theta_alt_motif)
    raw_motif_dev_b = pt.abs(p_alt_motifs_b - theta_alt_motif)
    smooth_motif_dev_a = (n_alt_motifs_a_f / (n_alt_motifs_a_f + K_motif)) * raw_motif_dev_a
    smooth_motif_dev_b = (n_alt_motifs_b_f / (n_alt_motifs_b_f + K_motif)) * raw_motif_dev_b

    # Score sequences based on their smoothed distance from the ideal (higher distance = lower score)
    score_a = -(balance_weight * smooth_imb_a + alt_weight * smooth_alt_dev_a + motif_weight * smooth_motif_dev_a)
    score_b = -(balance_weight * smooth_imb_b + alt_weight * smooth_alt_dev_b + motif_weight * smooth_motif_dev_b)

    # Softmax decision rule
    p_left_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
