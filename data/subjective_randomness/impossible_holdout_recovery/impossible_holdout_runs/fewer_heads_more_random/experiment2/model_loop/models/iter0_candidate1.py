"""
People judge the randomness of a sequence by its Bayesian diagnosticity for a fair coin, specifically contrasting it against an alternative hypothesis that the generative process is biased towards heads. They evaluate the log-likelihood ratio of the sequence under a fair coin versus a heads-biased coin, judging sequences that provide stronger evidence against the heads bias as more random.
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
    tau = pm.HalfNormal("tau", sigma=1.0)
    # The probability of heads under the alternative heads-biased model
    # Bound to ensure it represents a bias towards heads and avoids log(0)
    theta_biased = pm.Uniform("theta_biased", lower=0.51, upper=0.99)
    
    # Log-likelihood under a fair coin
    log_p_fair_a = n_a * pt.log(0.5)
    log_p_fair_b = n_b * pt.log(0.5)
    
    # Log-likelihood under the heads-biased alternative coin
    log_p_biased_a = h_a * pt.log(theta_biased) + (n_a - h_a) * pt.log(1.0 - theta_biased)
    log_p_biased_b = h_b * pt.log(theta_biased) + (n_b - h_b) * pt.log(1.0 - theta_biased)
    
    # Evidence for fair coin vs heads-biased coin (Bayesian diagnosticity)
    ev_a = log_p_fair_a - log_p_biased_a
    ev_b = log_p_fair_b - log_p_biased_b
    
    # Probability of choosing left (A)
    p_left_raw = pm.math.sigmoid(tau * (ev_a - ev_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
