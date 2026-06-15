import numpy as np
import pytensor.tensor as pt

def kl_div_from_fair(k, n):
    n_f = pt.cast(n, "float64")
    k_f = pt.cast(k, "float64")
    p = k_f / pt.maximum(n_f, 1.0)
    p = pt.clip(p, 1e-5, 1.0 - 1e-5)
    
    # KL(p || 0.5) = p * log(p / 0.5) + (1-p) * log((1-p) / 0.5)
    kl = p * pt.log(p * 2.0) + (1.0 - p) * pt.log((1.0 - p) * 2.0)
    
    # If n == 0 (e.g. n_trans for length 1), KL is 0
    return pt.switch(pt.gt(n_f, 0.0), kl, 0.0)

# Try compile
import pymc as pm
with pm.Model() as m:
    n = pm.Data("n", np.array([10]))
    k = pm.Data("k", np.array([3]))
    kl = kl_div_from_fair(k, n)
