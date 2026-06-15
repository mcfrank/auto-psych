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
    
    # We refine iter0_candidate2 by applying the cognitive constraint that 
    # structural regularities strictly penalize the subjective randomness score.
    # We keep the heavy-tailed StudentT prior which performed well, but constrain 
    # it to be strictly positive (HalfStudentT), halving the prior volume per 
    # parameter to improve model evidence.
    nu_val = 4.0
    w_run_pen = pm.HalfStudentT("w_run_pen", nu=nu_val, sigma=5.0)
    w_period_pen = pm.HalfStudentT("w_period_pen", nu=nu_val, sigma=5.0)
    w_imbalance_pen = pm.HalfStudentT("w_imbalance_pen", nu=nu_val, sigma=5.0)
    
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)
    
    # Structural features strictly reduce the subjective randomness score.
    score_a = -(w_run_pen * max_run_norm_a + w_period_pen * periodicity_a + w_imbalance_pen * imbalance_a)
    score_b = -(w_run_pen * max_run_norm_b + w_period_pen * periodicity_b + w_imbalance_pen * imbalance_b)
    
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid((score_a - score_b) + side_bias)
    )
    
    pm.Bernoulli("response", p=p_left, observed=chose_left)
