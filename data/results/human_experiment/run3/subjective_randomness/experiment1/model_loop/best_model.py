"""People judge the randomness of a sequence by comparing its features (head proportion and alternation rate) to a subjective ideal, but their psychological penalty for deviations is scaled by the square root of the sequence length, reflecting an intuitive sensitivity to the standard error of small samples. This statistical-evidence weighting means that while people still penalize non-ideal proportions, they are far more tolerant of extreme imbalance in very short sequences where the small sample size provides weak evidence of non-randomness."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Read the necessary precomputed columns
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.1, upper=20.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight

    # Scale absolute deviations by the square root of the relevant sample size
    sqrt_n_a = pt.sqrt(pt.cast(n_a, "float64"))
    sqrt_n_b = pt.sqrt(pt.cast(n_b, "float64"))
    
    n_alts_a = pt.maximum(n_a - 1, 1)
    n_alts_b = pt.maximum(n_b - 1, 1)
    sqrt_n_alts_a = pt.sqrt(pt.cast(n_alts_a, "float64"))
    sqrt_n_alts_b = pt.sqrt(pt.cast(n_alts_b, "float64"))

    score_a = -(
        balance_weight * sqrt_n_a * imbalance_a + 
        alt_weight * sqrt_n_alts_a * pt.abs(p_alts_a - theta_alt)
    )
    
    score_b = -(
        balance_weight * sqrt_n_b * imbalance_b + 
        alt_weight * sqrt_n_alts_b * pt.abs(p_alts_b - theta_alt)
    )

    p_left_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
