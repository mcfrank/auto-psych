"""
People judge sequence randomness based on the statistical evidence for outcome bias, perceiving sequences as more random when the log-likelihood ratio of a biased coin model versus a fair coin model is higher.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def log_lr_biased_vs_fair(h, n):
    # Safe division to prevent NaNs if n is 0
    safe_n = pt.clip(n, 1.0, 10000.0)
    p = pt.clip(h / safe_n, 1e-6, 1.0 - 1e-6)
    
    ll_biased = h * pt.log(p) + (n - h) * pt.log(1.0 - p)
    ll_fair = n * pt.log(0.5)
    return ll_biased - ll_fair

with pm.Model() as model:
    # Stimulus inputs 
    n_a = pm.Data("n_a", np.zeros(1, dtype="float64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="float64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="float64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter with a prior
    weight = pm.Normal("weight", mu=0.0, sigma=1.0)

    llr_a = log_lr_biased_vs_fair(h_a, n_a)
    llr_b = log_lr_biased_vs_fair(h_b, n_b)

    # Response probability
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(weight * (llr_a - llr_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
