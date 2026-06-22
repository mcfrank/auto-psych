"""
People judge the randomness of a sequence based on its Gaussian similarity to an ideal subjective prototype, meaning the sequence's perceived randomness decays exponentially with its squared distance in feature space (proportion of heads and alternations) from the expected prototype. This formulation penalizes extreme deviations more severely than a linear distance metric.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    ideal_p_heads = pm.Beta("ideal_p_heads", alpha=5.0, beta=5.0)
    ideal_p_alts = pm.Beta("ideal_p_alts", alpha=6.0, beta=4.0)
    
    precision_heads = pm.HalfNormal("precision_heads", sigma=10.0)
    precision_alts = pm.HalfNormal("precision_alts", sigma=10.0)
    
    # Squared Euclidean distance (weighted)
    dist_sq_a = precision_heads * (p_a - ideal_p_heads)**2 + precision_alts * (p_alts_a - ideal_p_alts)**2
    dist_sq_b = precision_heads * (p_b - ideal_p_heads)**2 + precision_alts * (p_alts_b - ideal_p_alts)**2
    
    # We choose the one that is closer to the prototype (smaller distance)
    # p_left is probability of choosing A
    p_left_raw = pm.math.sigmoid(dist_sq_b - dist_sq_a)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
