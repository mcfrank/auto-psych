"""Observers evaluate the randomness of a sequence by computing the true log Bayes factor between a fair-coin null and a biased-coin alternative, marginalizing over possible alternative biases using a subjective Beta prior instead of comparing to a single fixed bias."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    alpha = pm.HalfNormal("alpha", sigma=5.0)
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Safe lower bound for alpha to avoid log(0) in gammaln
    alpha_safe = pt.clip(alpha, 1e-4, 1e4)

    # Cast inputs to float64
    n_a_f = pt.cast(n_a, "float64")
    h_a_f = pt.cast(h_a, "float64")
    n_b_f = pt.cast(n_b, "float64")
    h_b_f = pt.cast(h_b, "float64")

    # Fair coin log likelihood
    log_fair_a = n_a_f * pt.log(0.5)
    log_fair_b = n_b_f * pt.log(0.5)

    # Log marginal likelihood under H1 (Beta(alpha, alpha) prior)
    def betaln(x, y):
        return pt.gammaln(x) + pt.gammaln(y) - pt.gammaln(x + y)

    log_bias_a = betaln(h_a_f + alpha_safe, n_a_f - h_a_f + alpha_safe) - betaln(alpha_safe, alpha_safe)
    log_bias_b = betaln(h_b_f + alpha_safe, n_b_f - h_b_f + alpha_safe) - betaln(alpha_safe, alpha_safe)

    # Log Bayes factor (evidence for fair coin over biased alternative)
    lbf_a = log_fair_a - log_bias_a
    lbf_b = log_fair_b - log_bias_b

    # Choice probability: prefer sequence with stronger evidence for fairness
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (lbf_a - lbf_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
