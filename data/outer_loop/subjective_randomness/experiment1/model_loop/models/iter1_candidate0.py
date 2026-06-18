"""People evaluate a sequence's randomness by comparing its likelihood under a fair coin against its probability under a biased coin, mathematically marginalizing over all possible alternative biases rather than assuming a single fixed bias. They hold a symmetric prior belief about the alternative coin's bias, with the concentration of this prior acting as a subjective parameter, and they prefer the sequence that provides stronger Bayesian evidence for the fair coin."""

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
    # The concentration parameter of the symmetric Beta prior. Add a small constant for numerical stability.
    alpha_base = pm.HalfNormal("alpha_base", sigma=5.0)
    alpha = alpha_base + 1e-4
    
    # Softmax temperature
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Cast to float for computations
    n_a_f = pt.cast(n_a, "float64")
    h_a_f = pt.cast(h_a, "float64")
    n_b_f = pt.cast(n_b, "float64")
    h_b_f = pt.cast(h_b, "float64")

    # Compute exact log marginal likelihood for the biased coin (Beta-Binomial)
    # log P(D | biased) = log[ B(h + alpha, n - h + alpha) / B(alpha, alpha) ]
    # where B(x, y) = Gamma(x)Gamma(y) / Gamma(x + y)
    def log_marginal_biased(h_f, n_f, a):
        log_beta_post = pt.gammaln(h_f + a) + pt.gammaln(n_f - h_f + a) - pt.gammaln(n_f + 2.0 * a)
        log_beta_prior = 2.0 * pt.gammaln(a) - pt.gammaln(2.0 * a)
        return log_beta_post - log_beta_prior

    log_bias_a = log_marginal_biased(h_a_f, n_a_f, alpha)
    log_bias_b = log_marginal_biased(h_b_f, n_b_f, alpha)

    # Compute log likelihood for the fair coin
    log_fair_a = n_a_f * pt.log(0.5)
    log_fair_b = n_b_f * pt.log(0.5)

    # Log Bayes factor in favor of the fair coin
    lbf_a = log_fair_a - log_bias_a
    lbf_b = log_fair_b - log_bias_b

    # Choice probability
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (lbf_a - lbf_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)

