import numpy as np
import pymc as pm
import pytensor.tensor as pt

"""
Random-looking sequences are judged by an evidence accumulation process where each item provides a baseline weight of evidence, which is discounted by the sequence's quadratic deviation from a three-dimensional messy prototype. This prototype consists of an ideal positive imbalance, an ideal alternation rate, and an ideal normalized maximum run length, directly penalizing sequences that contain disproportionately long local streaks even when their global alternation rate is typical.
"""

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    baseline_evidence = pm.Normal("baseline_evidence", mu=0.0, sigma=1.0)
    
    # Ideal prototype features
    ideal_imbalance = pm.HalfNormal("ideal_imbalance", sigma=0.5)
    ideal_p_alts = pm.Beta("ideal_p_alts", alpha=2.0, beta=2.0)
    ideal_max_run = pm.Beta("ideal_max_run", alpha=2.0, beta=2.0)

    # Weights for the penalty
    w_imbalance = pm.HalfNormal("w_imbalance", sigma=5.0)
    w_alts = pm.HalfNormal("w_alts", sigma=5.0)
    w_max_run = pm.HalfNormal("w_max_run", sigma=5.0)
    
    # Decision noise
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Evidence calculation: baseline times sequence length, minus weighted quadratic deviations
    ev_a = n_a * baseline_evidence \
           - w_imbalance * (imbalance_a - ideal_imbalance)**2 \
           - w_alts * (p_alts_a - ideal_p_alts)**2 \
           - w_max_run * (max_run_norm_a - ideal_max_run)**2

    ev_b = n_b * baseline_evidence \
           - w_imbalance * (imbalance_b - ideal_imbalance)**2 \
           - w_alts * (p_alts_b - ideal_p_alts)**2 \
           - w_max_run * (max_run_norm_b - ideal_max_run)**2

    # Logistic choice rule based on difference in evidence
    # Clip probabilities for numerical safety
    p_left_raw = pm.math.sigmoid(tau * (ev_a - ev_b))
    p_left_clipped = pt.clip(p_left_raw, 1e-6, 1 - 1e-6)
    p_left = pm.Deterministic("p_left", p_left_clipped)

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
