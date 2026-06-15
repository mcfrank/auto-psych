import numpy as np
import pymc as pm

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    
    # Cognitive principle: participants evaluate compressibility using a convex 
    # combination of penalty features, scaled by a global precision/temperature parameter.
    longrun_weight = pm.Beta("longrun_weight", alpha=1.0, beta=1.0)
    periodic_share = pm.Beta("periodic_share", alpha=1.0, beta=1.0)
    
    w_run = longrun_weight
    w_period = (1.0 - longrun_weight) * periodic_share
    w_imbalance = (1.0 - longrun_weight) * (1.0 - periodic_share)
    
    beta = pm.HalfNormal("beta", sigma=10.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)
    
    # Negative weight means the feature penalizes the randomness score.
    score_a = -(w_run * max_run_norm_a + w_period * periodicity_a + w_imbalance * imbalance_a)
    score_b = -(w_run * max_run_norm_b + w_period * periodicity_b + w_imbalance * imbalance_b)
    
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    
    # Observed response: the pm.Data tensor is passed directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
