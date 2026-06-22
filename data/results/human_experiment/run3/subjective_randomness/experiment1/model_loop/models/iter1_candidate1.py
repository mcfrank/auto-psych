"""People judge the randomness of a sequence by comparing it to an ideal prototype, but they evaluate its departure from this ideal using the absolute difference in the raw counts of heads and alternations, rather than their proportions. By using unnormalized counts, people naturally exhibit length-dependent tolerance, applying much smaller penalties to perfectly imbalanced short sequences than to perfectly imbalanced long sequences."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Features
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.1, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight

    # Absolute difference in counts.
    expected_h_a = n_a / 2.0
    imbalance_count_a = pt.abs(h_a - expected_h_a)

    expected_alts_a = (n_a - 1.0) * theta_alt
    alts_diff_count_a = pt.abs(alts_a - expected_alts_a)

    expected_h_b = n_b / 2.0
    imbalance_count_b = pt.abs(h_b - expected_h_b)

    expected_alts_b = (n_b - 1.0) * theta_alt
    alts_diff_count_b = pt.abs(alts_b - expected_alts_b)

    score_a = -(balance_weight * imbalance_count_a + alt_weight * alts_diff_count_a)
    score_b = -(balance_weight * imbalance_count_b + alt_weight * alts_diff_count_b)

    p_left_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
