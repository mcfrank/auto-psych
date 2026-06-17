"""
People evaluate the randomness of a sequence solely by checking whether its proportion of heads matches the expectation of a fair coin (0.5). Sequences whose proportion of heads is closer to 0.5 are consistently judged as being more representative of a random process.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    
    # Free cognitive parameter: temperature for the choice rule
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    # Calculate proportion of heads, protecting against division by zero
    len_a = pt.maximum(n_a, 1)
    len_b = pt.maximum(n_b, 1)
    prop_a = h_a / len_a
    prop_b = h_b / len_b
    
    # Absolute distance from the expected fair coin proportion (0.5)
    dist_a = pt.abs(prop_a - 0.5)
    dist_b = pt.abs(prop_b - 0.5)
    
    # Smaller distance to 0.5 means it is judged more random.
    # Therefore, A is chosen if dist_a < dist_b, so score_diff is dist_b - dist_a.
    score_diff = dist_b - dist_a
    
    # Choice probability with numerical safety
    p_left = pm.math.sigmoid(tau * score_diff)
    p_left_safe = pm.Deterministic("p_left", pt.clip(p_left, 1e-6, 1 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
