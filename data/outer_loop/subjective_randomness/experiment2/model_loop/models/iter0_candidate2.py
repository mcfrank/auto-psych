"""
Observers evaluate the randomness of a sequence using a point-estimate log-likelihood ratio rather than full Bayesian integration. They estimate the sequence's bias by smoothing the empirical proportion of heads with subjective pseudo-counts, and judge the sequence as less random the more its likelihood under this estimated bias exceeds its likelihood under a fair coin.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    
    # Cognitive parameters
    alpha = pm.HalfNormal("alpha", sigma=2.0)  # Subjective pseudo-counts
    tau = pm.HalfNormal("tau", sigma=1.0)      # Decision noise
    
    # Compute smoothed empirical proportions (point estimates of bias)
    # Clipped for absolute numerical safety against log(0) if alpha -> 0
    p_a_emp = pt.clip((h_a + alpha) / (n_a + 2.0 * alpha), 1e-5, 1.0 - 1e-5)
    p_b_emp = pt.clip((h_b + alpha) / (n_b + 2.0 * alpha), 1e-5, 1.0 - 1e-5)
    
    # Log-likelihood under the estimated bias
    ll_biased_a = h_a * pt.log(p_a_emp) + (n_a - h_a) * pt.log(1.0 - p_a_emp)
    ll_biased_b = h_b * pt.log(p_b_emp) + (n_b - h_b) * pt.log(1.0 - p_b_emp)
    
    # Log-likelihood under a fair coin (p = 0.5)
    ll_fair_a = n_a * pt.log(0.5)
    ll_fair_b = n_b * pt.log(0.5)
    
    # Log-likelihood ratio (randomness score)
    # LLR is <= 0. Higher (closer to 0) means the sequence is more random.
    llr_a = ll_fair_a - ll_biased_a
    llr_b = ll_fair_b - ll_biased_b
    
    # Choice probability
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (llr_a - llr_b)))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
