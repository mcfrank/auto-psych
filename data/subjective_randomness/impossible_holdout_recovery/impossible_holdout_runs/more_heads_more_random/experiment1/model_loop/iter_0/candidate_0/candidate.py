"""
People judge the randomness of a sequence by its similarity to a prototype, but rather than expecting balanced outcomes, their prototype expects a sequence to be dominated by heads. Sequences with a higher proportion of heads are perceived as closer to the prototype and therefore more random.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    
    # Free parameters
    ideal_heads = pm.Beta("ideal_heads", alpha=1.0, beta=1.0)
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # Distance to the heads-biased prototype
    dist_a = pt.abs(p_a - ideal_heads)
    dist_b = pt.abs(p_b - ideal_heads)
    
    # Probability of choosing A (A is more random if distance is smaller)
    p_left_raw = pm.math.sigmoid(tau * (dist_b - dist_a))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
