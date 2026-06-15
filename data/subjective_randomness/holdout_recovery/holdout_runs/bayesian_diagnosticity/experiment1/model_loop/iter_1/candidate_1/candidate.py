"""PyMC model implementing a Gaussian Prototype (Tolerance) mechanism.

Unlike linear/Laplace prototype models, this assumes people have a 'tolerance window'
for deviations from randomness. Small deviations from ideal properties are largely ignored, 
but perception of non-randomness grows quadratically for salient deviations.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs (names must match responses.csv columns)
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Prototype ideal alternation rate (gambler's fallacy suggests ~0.6)
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    
    # Feature weights (HalfNormal acts as both weight and inverse-temperature)
    # This avoids unidentifiability between a global beta and individual weights.
    w_imb = pm.HalfNormal("w_imb", sigma=10.0)
    w_alt = pm.HalfNormal("w_alt", sigma=10.0)
    w_run = pm.HalfNormal("w_run", sigma=10.0)
    w_per = pm.HalfNormal("w_per", sigma=10.0)
    
    # Choice parameters
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)

    # Quadratic penalty (Euclidean distance to mental prototype)
    # Larger scores mean the sequence is perceived as LESS random
    penalty_a = (
        w_imb * pt.square(imbalance_a) +
        w_alt * pt.square(p_alts_a - theta_alt) +
        w_run * pt.square(max_run_norm_a) +
        w_per * pt.square(periodicity_a)
    )
    
    penalty_b = (
        w_imb * pt.square(imbalance_b) +
        w_alt * pt.square(p_alts_b - theta_alt) +
        w_run * pt.square(max_run_norm_b) +
        w_per * pt.square(periodicity_b)
    )

    # Probability of choosing sequence A (left)
    # If A has a lower penalty (more random), penalty_b - penalty_a is positive -> p_left > 0.5
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid((penalty_b - penalty_a) + side_bias)
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
