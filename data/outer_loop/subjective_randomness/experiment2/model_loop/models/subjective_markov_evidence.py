"""Observers evaluate randomness by computing the log Bayes factor of the sequence's transitions under a subjective ideal Markov process versus a purely independent fair coin, preferring sequences that provide stronger evidence for their subjective ideal."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    theta = pm.Beta("theta", alpha=2.0, beta=2.0)     # ideal transition probability
    tau = pm.HalfNormal("tau", sigma=2.0)             # softmax temperature

    theta_safe = pt.clip(theta, 1e-6, 1.0 - 1e-6)

    # Log likelihood under subjective Markov model (the "random" concept)
    log_ideal_a = pt.cast(alts_a, "float64") * pt.log(theta_safe) + pt.cast(n_a - 1 - alts_a, "float64") * pt.log(1.0 - theta_safe)
    # Log likelihood under independent fair coin (the "null")
    log_fair_a = pt.cast(n_a - 1, "float64") * pt.log(0.5)
    
    # Evidence for subjective ideal over fair coin
    lbf_a = log_ideal_a - log_fair_a

    log_ideal_b = pt.cast(alts_b, "float64") * pt.log(theta_safe) + pt.cast(n_b - 1 - alts_b, "float64") * pt.log(1.0 - theta_safe)
    log_fair_b = pt.cast(n_b - 1, "float64") * pt.log(0.5)
    
    lbf_b = log_ideal_b - log_fair_b

    # Prefer the sequence with stronger evidence for the subjective ideal
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (lbf_a - lbf_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
