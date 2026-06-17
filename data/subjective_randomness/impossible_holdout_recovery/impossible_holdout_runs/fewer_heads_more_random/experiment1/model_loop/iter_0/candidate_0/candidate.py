"""
People judge randomness by comparing the likelihood of the sequence under a fair coin to its likelihood under a single Markov alternative model that generates alternations at a non-random rate. The transition probability of this alternative model is a free cognitive parameter.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    
    # Cognitive parameters
    # Transition probability for the alternative Markov model
    theta_alt = pm.Beta("theta_alt", alpha=2.0, beta=2.0)
    
    # Scaling parameter for the decision
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # Ensure numerical safety for theta_alt in log
    theta_alt_safe = pt.clip(theta_alt, 1e-6, 1.0 - 1e-6)
    
    # Calculate number of transitions
    trans_n_a = pt.maximum(n_a - 1, 0)
    trans_n_b = pt.maximum(n_b - 1, 0)
    
    # Log-likelihood under a fair coin (transition probability = 0.5)
    ll_fair_a = trans_n_a * pt.log(0.5)
    ll_fair_b = trans_n_b * pt.log(0.5)
    
    # Log-likelihood under the alternative Markov model
    ll_alt_a = alts_a * pt.log(theta_alt_safe) + (trans_n_a - alts_a) * pt.log(1.0 - theta_alt_safe)
    ll_alt_b = alts_b * pt.log(theta_alt_safe) + (trans_n_b - alts_b) * pt.log(1.0 - theta_alt_safe)
    
    # Evidence for fair coin over the alternative
    evidence_a = ll_fair_a - ll_alt_a
    evidence_b = ll_fair_b - ll_alt_b
    
    # Decision rule: softmax over the evidence
    p_left_val = pm.math.sigmoid(tau * (evidence_a - evidence_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_val, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
