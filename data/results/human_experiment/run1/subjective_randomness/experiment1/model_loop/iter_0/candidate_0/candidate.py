"""
People judge randomness by comparing a sequence to a mental prototype, but this prototype is subjectively biased: it possesses an ideal proportion of heads and an ideal alternation rate that may deviate from objective fairness. Sequences are perceived as more random when their proportion of heads and alternations have a smaller squared deviation from these subjective ideals.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Features from responses
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Free parameters for the subjective prototype
    ideal_p = pm.Beta("ideal_p", alpha=2.0, beta=2.0)
    ideal_alt = pm.Beta("ideal_alt", alpha=2.0, beta=2.0)
    
    # Weights for the deviations
    w_p = pm.HalfNormal("w_p", sigma=5.0)
    w_alt = pm.HalfNormal("w_alt", sigma=5.0)
    
    # Calculate empirical rates (safeguard against division by zero)
    p_a = h_a / pt.maximum(n_a, 1)
    p_b = h_b / pt.maximum(n_b, 1)
    
    alt_rate_a = alts_a / pt.maximum(n_a - 1, 1)
    alt_rate_b = alts_b / pt.maximum(n_b - 1, 1)
    
    # Calculate squared deviations from the subjective ideals
    dev_a = w_p * pt.square(p_a - ideal_p) + w_alt * pt.square(alt_rate_a - ideal_alt)
    dev_b = w_p * pt.square(p_b - ideal_p) + w_alt * pt.square(alt_rate_b - ideal_alt)
    
    # Lower deviation means it is closer to the prototype (more random)
    # Using sigmoid of (dev_b - dev_a) so that if dev_a < dev_b, p_left > 0.5
    # Clip to avoid exact 0 or 1 for numerical stability
    p_left_raw = pm.math.sigmoid(dev_b - dev_a)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
