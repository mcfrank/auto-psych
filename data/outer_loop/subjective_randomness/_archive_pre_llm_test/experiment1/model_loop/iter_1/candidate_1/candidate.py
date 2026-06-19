"""Observers evaluate a sequence's randomness by computing the log Bayes factor between an independent fair coin and an alternative first-order Markov process with a fixed, subjective transition probability."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    theta = pm.Beta("theta", alpha=2.0, beta=2.0)  # Transition probability of the Markov alternative
    tau = pm.HalfNormal("tau", sigma=2.0)          # Softmax temperature

    # Avoid log(0)
    theta_safe = pt.clip(theta, 1e-6, 1.0 - 1e-6)

    # Number of transitions
    t_a = pt.cast(n_a, "float64") - 1.0
    t_b = pt.cast(n_b, "float64") - 1.0
    
    alts_a_f = pt.cast(alts_a, "float64")
    alts_b_f = pt.cast(alts_b, "float64")

    # Log likelihoods for Sequence A (transitions only)
    log_fair_a = t_a * pt.log(0.5)
    log_markov_a = alts_a_f * pt.log(theta_safe) + (t_a - alts_a_f) * pt.log(1.0 - theta_safe)
    lbf_a = log_fair_a - log_markov_a

    # Log likelihoods for Sequence B (transitions only)
    log_fair_b = t_b * pt.log(0.5)
    log_markov_b = alts_b_f * pt.log(theta_safe) + (t_b - alts_b_f) * pt.log(1.0 - theta_safe)
    lbf_b = log_fair_b - log_markov_b

    # Probability of choosing sequence A (left)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (lbf_a - lbf_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
