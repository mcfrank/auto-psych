"""People judge a sequence's randomness by evaluating the joint typicality of its macroscopic features — specifically, the binomial probability of its head count given a fair coin, and the binomial probability of its alternation count given an ideal alternation rate. Rather than using length-invariant proportional heuristics, this mechanism naturally scales deviation penalties with sequence length and provides a statistically grounded, asymmetric tolerance for over-alternating (highly periodic) sequences."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def log_binom_prob(n, k, p):
    """Compute the log probability of k successes in n trials with probability p."""
    n_f = pt.cast(n, "float64")
    k_f = pt.cast(k, "float64")
    # Clamp p to avoid log(0)
    p_safe = pt.clip(p, 1e-6, 1.0 - 1e-6)
    log_comb = (
        pt.gammaln(n_f + 1.0) - pt.gammaln(k_f + 1.0) - pt.gammaln(n_f - k_f + 1.0)
    )
    return log_comb + k_f * pt.log(p_safe) + (n_f - k_f) * pt.log(1.0 - p_safe)


with pm.Model() as model:
    # Stimulus inputs (macroscopic counts)
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight

    # Number of possible alternations is n - 1. We clip to 1 to prevent n=1 edge cases.
    N_a = pt.maximum(n_a - 1, 1)
    N_b = pt.maximum(n_b - 1, 1)

    # Calculate typicality scores as weighted joint log-probabilities
    # log P(heads | fair coin)
    log_p_h_a = log_binom_prob(n_a, h_a, 0.5)
    log_p_h_b = log_binom_prob(n_b, h_b, 0.5)

    # log P(alternations | ideal alternation rate)
    log_p_alt_a = log_binom_prob(N_a, alts_a, theta_alt)
    log_p_alt_b = log_binom_prob(N_b, alts_b, theta_alt)

    score_a = balance_weight * log_p_h_a + alt_weight * log_p_alt_a
    score_b = balance_weight * log_p_h_b + alt_weight * log_p_alt_b

    # Choice probability with a psychometric function
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )

    pm.Bernoulli("response", p=p_left, observed=chose_left)
