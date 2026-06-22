"""
People judge the randomness of a sequence by the probabilistic surprise of its macroscopic features (such as heads and alternations), rather than the likelihood of the specific sequence. They intuitively evaluate the exact Binomial probability of observing those specific feature counts under a subjective ideal model, naturally perceiving sequences with highly probable feature counts as more random while appropriately penalizing deviations more strictly in longer sequences.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Subjective ideals and parameters
    ideal_alts = pm.Beta("ideal_alts", alpha=5.0, beta=5.0)
    w_alts = pm.HalfNormal("w_alts", sigma=2.0)
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Calculate log-probabilities of features under the subjective ideal model
    # Probability of heads (ideal p=0.5)
    log_p_h_a = pm.logp(pm.Binomial.dist(n=n_a, p=0.5), h_a)
    log_p_h_b = pm.logp(pm.Binomial.dist(n=n_b, p=0.5), h_b)

    # Probability of alternations (number of possible alternations is n-1)
    # Use maximum to avoid invalid n=0 binomial parameter during dummy initialization
    n_alts_a = pm.math.maximum(1, n_a - 1)
    n_alts_b = pm.math.maximum(1, n_b - 1)

    log_p_alts_a = pm.logp(pm.Binomial.dist(n=n_alts_a, p=ideal_alts), alts_a)
    log_p_alts_b = pm.logp(pm.Binomial.dist(n=n_alts_b, p=ideal_alts), alts_b)

    # Total score is the weighted sum of log probabilities (higher is more random)
    score_a = log_p_h_a + w_alts * log_p_alts_a
    score_b = log_p_h_b + w_alts * log_p_alts_b

    # Probability of choosing 'left' (sequence a)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
