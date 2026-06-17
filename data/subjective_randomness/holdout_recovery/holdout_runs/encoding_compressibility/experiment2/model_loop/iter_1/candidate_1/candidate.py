import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    
    # Weights for the different encoding heuristics
    w_run = pm.HalfNormal("w_run", sigma=5.0)
    w_period = pm.HalfNormal("w_period", sigma=5.0)
    w_imbalance = pm.HalfNormal("w_imbalance", sigma=5.0)
    
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=5.0)
    
    # Cognitive mechanism: Minimum Description Length (MDL) approximation.
    # Participants notice the *most prominent* pattern (the one that compresses the most).
    # The penalty for a sequence is the maximum of the penalties from different encoding strategies.
    penalty_a = pt.maximum(
        w_run * max_run_norm_a,
        pt.maximum(w_period * periodicity_a, w_imbalance * imbalance_a)
    )
    
    penalty_b = pt.maximum(
        w_run * max_run_norm_b,
        pt.maximum(w_period * periodicity_b, w_imbalance * imbalance_b)
    )
    
    # Less compressible = more random. So score is negative penalty.
    score_a = -penalty_a
    score_b = -penalty_b
    
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    
    # Observed response: the pm.Data tensor is passed directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
