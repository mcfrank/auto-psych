"""
People judge the randomness of a sequence by its smoothed distance from a subjective prototype, but this prototype tracks the proportion of 4-item alternating motifs (like HTHT) instead of simple bigram alternations, viewing these longer, more complex alternations as the primary structural signature of local randomness.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Feature inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))

    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    alt_motifs_a = pm.Data("alt_motifs_a", np.zeros(1, dtype="int64"))
    alt_motifs_b = pm.Data("alt_motifs_b", np.zeros(1, dtype="int64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Cognitive parameters
    theta_alt_motif = pm.Beta("theta_alt_motif", alpha=1.0, beta=1.0)

    balance_weight = pm.Beta("balance_weight", alpha=1.0, beta=1.0)
    motif_weight = 1.0 - balance_weight

    K_bal = pm.HalfNormal("K_bal", sigma=10.0)
    K_motif = pm.HalfNormal("K_motif", sigma=10.0)

    beta = pm.HalfNormal("beta", sigma=10.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # Calculations
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")

    # Number of valid 4-item motifs
    n_motifs_a_f = pt.cast(pt.maximum(n_a - 3, 0), "float64")
    n_motifs_b_f = pt.cast(pt.maximum(n_b - 3, 0), "float64")

    # Raw proportions (safe division: when n_motifs is 0, denominator is 1.0, alt_motifs is 0, so p is 0.0)
    p_alt_motifs_a = pt.cast(alt_motifs_a, "float64") / pt.maximum(n_motifs_a_f, 1.0)
    p_alt_motifs_b = pt.cast(alt_motifs_b, "float64") / pt.maximum(n_motifs_b_f, 1.0)

    # Smoothed deviations
    smooth_imb_a = (n_a_f / (n_a_f + K_bal + 1e-6)) * imbalance_a
    smooth_imb_b = (n_b_f / (n_b_f + K_bal + 1e-6)) * imbalance_b

    raw_motif_dev_a = pt.abs(p_alt_motifs_a - theta_alt_motif)
    raw_motif_dev_b = pt.abs(p_alt_motifs_b - theta_alt_motif)

    smooth_motif_dev_a = (
        n_motifs_a_f / (n_motifs_a_f + K_motif + 1e-6)
    ) * raw_motif_dev_a
    smooth_motif_dev_b = (
        n_motifs_b_f / (n_motifs_b_f + K_motif + 1e-6)
    ) * raw_motif_dev_b

    # Combine into a final distance score
    score_a = -(balance_weight * smooth_imb_a + motif_weight * smooth_motif_dev_a)
    score_b = -(balance_weight * smooth_imb_b + motif_weight * smooth_motif_dev_b)

    # Decision rule
    p_left_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
