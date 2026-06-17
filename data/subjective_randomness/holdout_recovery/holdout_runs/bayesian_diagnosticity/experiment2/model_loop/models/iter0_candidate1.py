"""Feature Heuristics Cognitive Model.

People judge sequence randomness based on a linear combination of heuristic features:
balance (deviation from 50%), alternation rate, longest run length, and periodicity.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs (pre-computed features)
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    w_imbalance = pm.Normal("w_imbalance", mu=0.0, sigma=3.0)
    w_alts = pm.Normal("w_alts", mu=0.0, sigma=3.0)
    w_max_run = pm.Normal("w_max_run", mu=0.0, sigma=3.0)
    w_periodicity = pm.Normal("w_periodicity", mu=0.0, sigma=3.0)
    
    # Preferred alternation rate (typically slightly above 0.5)
    theta_alt = pm.Uniform("theta_alt", lower=0.4, upper=0.9)
    
    # Decision noise and side bias
    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)

    # Score A: higher score means MORE likely to be judged as random.
    # Deviation from theta_alt is penalized.
    score_a = -(
        w_imbalance * imbalance_a +
        w_alts * pt.abs(p_alts_a - theta_alt) +
        w_max_run * max_run_norm_a +
        w_periodicity * periodicity_a
    )
    
    # Score B
    score_b = -(
        w_imbalance * imbalance_b +
        w_alts * pt.abs(p_alts_b - theta_alt) +
        w_max_run * max_run_norm_b +
        w_periodicity * periodicity_b
    )

    # Probability of choosing 'A' (left)
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
