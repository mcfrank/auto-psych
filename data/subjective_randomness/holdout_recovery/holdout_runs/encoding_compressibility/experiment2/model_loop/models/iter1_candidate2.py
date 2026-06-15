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
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    
    # Free parameters for the compressibility features
    w_run = pm.Normal("w_run", mu=0.0, sigma=3.0)
    w_period = pm.Normal("w_period", mu=0.0, sigma=3.0)
    w_imbalance = pm.Normal("w_imbalance", mu=0.0, sigma=3.0)
    
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)
    
    # Introduce a lapse rate to capture higher-variance/noisy responses
    # This acts as a robust likelihood for binary choices.
    # We use a Beta prior concentrated near low values but allowing for some lapses.
    lapse_rate = pm.Beta("lapse_rate", alpha=1.0, beta=9.0) 
    
    # Score calculation for each option
    score_a = w_run * max_run_norm_a + w_period * periodicity_a + w_imbalance * imbalance_a
    score_b = w_run * max_run_norm_b + w_period * periodicity_b + w_imbalance * imbalance_b
    
    # Core decision probability
    p_core = pm.math.sigmoid((score_a - score_b) + side_bias)
    
    # Mixture of the core decision process and random guessing (lapse)
    p_left = pm.Deterministic(
        "p_left",
        (1.0 - lapse_rate) * p_core + lapse_rate * 0.5
    )
    
    # Observed response
    pm.Bernoulli("response", p=p_left, observed=chose_left)
