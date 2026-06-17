import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Sequences are evaluated based on features of their shortest mental description.
    # We load the normalized structural features.
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    
    # We hypothesize that observers have a limited attention/weight capacity 
    # that they distribute among different compression heuristics.
    # We use a stick-breaking parameterization to ensure weights are strictly 
    # positive and sum to 1, providing strong regularization.
    longrun_weight = pm.Beta("longrun_weight", alpha=2.0, beta=2.0)
    periodic_share = pm.Beta("periodic_share", alpha=2.0, beta=2.0)
    
    # Overall sensitivity to subjective randomness differences
    beta = pm.HalfNormal("beta", sigma=5.0)
    
    # Task-specific side bias
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)
    
    # Compute the effective fractional weight for each heuristic
    w_run = longrun_weight
    remaining = 1.0 - longrun_weight
    w_period = remaining * periodic_share
    w_imbalance = remaining * (1.0 - periodic_share)
    
    # Compressibility penalty: higher penalty means the sequence is MORE compressible, 
    # and therefore LESS subjectively random.
    penalty_a = w_run * max_run_norm_a + w_period * periodicity_a + w_imbalance * imbalance_a
    penalty_b = w_run * max_run_norm_b + w_period * periodicity_b + w_imbalance * imbalance_b
    
    # Subjective randomness score is the negative of the compressibility penalty
    score_a = -penalty_a
    score_b = -penalty_b
    
    # Probability of choosing sequence A (left) as more random
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    
    pm.Bernoulli("response", p=p_left, observed=chose_left)
