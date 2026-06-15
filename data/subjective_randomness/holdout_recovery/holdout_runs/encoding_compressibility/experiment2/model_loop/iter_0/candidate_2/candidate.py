import numpy as np
import pymc as pm

with pm.Model() as model:
    # Read the features
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    
    # Higher variance / heavy-tailed prior for the parameters
    # This allows for larger weights and more extreme probability differences
    # without aggressively shrinking them to zero.
    nu_val = 4.0
    w_run = pm.StudentT("w_run", nu=nu_val, mu=0.0, sigma=5.0)
    w_period = pm.StudentT("w_period", nu=nu_val, mu=0.0, sigma=5.0)
    w_imbalance = pm.StudentT("w_imbalance", nu=nu_val, mu=0.0, sigma=5.0)
    
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)
    
    # Score calculation
    score_a = w_run * max_run_norm_a + w_period * periodicity_a + w_imbalance * imbalance_a
    score_b = w_run * max_run_norm_b + w_period * periodicity_b + w_imbalance * imbalance_b
    
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid((score_a - score_b) + side_bias)
    )
    
    pm.Bernoulli("response", p=p_left, observed=chose_left)
