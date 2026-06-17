"""PyMC adapter for the prototype-similarity model family, refined with a run length penalty."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    
    # Prototype parameters
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    
    # Feature weights (stick-breaking to sum to 1)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    run_share = pm.Uniform("run_share", lower=0.01, upper=0.99)
    
    remaining = 1.0 - alt_weight
    run_weight = remaining * run_share
    balance_weight = remaining * (1.0 - run_share)

    # Choice parameters
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Distance-based scores (closer to prototype = higher score)
    score_a = -(
        balance_weight * imbalance_a 
        + alt_weight * pt.abs(p_alts_a - theta_alt)
        + run_weight * max_run_norm_a
    )
    score_b = -(
        balance_weight * imbalance_b 
        + alt_weight * pt.abs(p_alts_b - theta_alt)
        + run_weight * max_run_norm_b
    )

    # Deterministic choice probability
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
