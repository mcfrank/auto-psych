# file: rational_markov.py
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus features
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    # tau: softmax temperature for the decision rule
    tau = pm.HalfNormal("tau", sigma=5.0)
    # theta_reg: the transition probability of the alternative "regular" Markov model
    theta_reg = pm.Beta("theta_reg", alpha=1.0, beta=1.0)

    # Clip theta_reg to avoid log(0)
    theta_reg_safe = pt.clip(theta_reg, 1e-5, 1.0 - 1e-5)

    # Log-likelihood of each sequence under the random model (fair coin)
    log_p_rand_a = n_a * pt.log(0.5)
    log_p_rand_b = n_b * pt.log(0.5)

    # Log-likelihood of each sequence under the regular model (Markov)
    # The first flip has probability 0.5, followed by (n-1) transitions.
    log_p_reg_a = (
        pt.log(0.5)
        + alts_a * pt.log(theta_reg_safe)
        + (n_a - 1 - alts_a) * pt.log(1.0 - theta_reg_safe)
    )
    log_p_reg_b = (
        pt.log(0.5)
        + alts_b * pt.log(theta_reg_safe)
        + (n_b - 1 - alts_b) * pt.log(1.0 - theta_reg_safe)
    )

    # Log-likelihood ratio (Random vs Regular)
    # Higher LLR means the sequence is more likely to be random than regular.
    llr_a = log_p_rand_a - log_p_reg_a
    llr_b = log_p_rand_b - log_p_reg_b

    # Probability of choosing sequence A (left) as more random
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (llr_a - llr_b)))

    # Observed responses
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
