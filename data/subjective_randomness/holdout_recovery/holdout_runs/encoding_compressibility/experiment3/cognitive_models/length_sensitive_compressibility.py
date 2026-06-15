import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # The best features from the inner loop model
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    
    # Heavy-tailed priors for features
    nu_val = 4.0
    w_run = pm.StudentT("w_run", nu=nu_val, mu=0.0, sigma=5.0)
    w_period = pm.StudentT("w_period", nu=nu_val, mu=0.0, sigma=5.0)
    w_imbalance = pm.StudentT("w_imbalance", nu=nu_val, mu=0.0, sigma=5.0)
    
    # Weight for sequence length
    w_length = pm.Normal("w_length", mu=0.0, sigma=2.0)
    
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)
    
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")
    
    score_a = w_run * max_run_norm_a + w_period * periodicity_a + w_imbalance * imbalance_a + w_length * n_a_f
    score_b = w_run * max_run_norm_b + w_period * periodicity_b + w_imbalance * imbalance_b + w_length * n_b_f
    
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid((score_a - score_b) + side_bias)
    )
    
    pm.Bernoulli("response", p=p_left, observed=chose_left)
