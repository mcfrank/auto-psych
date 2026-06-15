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
    
    n_a = pm.Data("n_a", np.zeros(1, dtype="float64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="float64"))

    # Cognitive parameters
    # Ideal alternation rate
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    
    # Feature weights summing to 1
    weights = pm.Dirichlet("weights", a=np.ones(3))
    weight_imbalance = weights[0]
    weight_alt = weights[1]
    weight_run = weights[2]
    
    # Evidence scaling: how much does sequence length magnify the penalty?
    # 0 = no scaling, 1 = linear scaling
    length_power = pm.Uniform("length_power", lower=0.0, upper=2.0)
    
    # Decision parameters
    beta = pm.Uniform("beta", lower=0.2, upper=20.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)

    # Scale distance by a power of sequence length
    ev_scale_a = pt.pow(n_a, length_power)
    ev_scale_b = pt.pow(n_b, length_power)
    
    # Penalize deviations from prototype
    penalty_a = ev_scale_a * (
        weight_imbalance * imbalance_a + 
        weight_alt * pt.abs(p_alts_a - theta_alt) +
        weight_run * max_run_norm_a
    )
    penalty_b = ev_scale_b * (
        weight_imbalance * imbalance_b + 
        weight_alt * pt.abs(p_alts_b - theta_alt) +
        weight_run * max_run_norm_b
    )

    # Deterministic probability of choosing left
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (penalty_b - penalty_a) + side_bias),
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
