"""
People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains. Their choice between sequences is driven by a cumulative normal (probit) discrimination process on the difference in head counts, reflecting Gaussian internal noise with no baseline rate of random guessing.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    
    # Free cognitive parameter with a prior
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # Calculate difference
    diff = pt.cast(h_a - h_b, "float64")
    
    # Probit link: CDF of the standard normal distribution
    p_raw = 0.5 + 0.5 * pt.erf(tau * diff / np.sqrt(2.0))
    
    # Numerical safety clamp to avoid log(0) in likelihood
    p_left = pm.Deterministic("p_left", pt.clip(p_raw, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
