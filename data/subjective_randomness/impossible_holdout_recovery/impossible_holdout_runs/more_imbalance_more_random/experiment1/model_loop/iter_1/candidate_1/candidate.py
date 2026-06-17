"""
People judge sequence randomness based on the Bayesian diagnosticity of a biased coin relative to a fair coin. Sequences that provide stronger evidence for a biased coin (which can favor either heads or tails) over a fair coin are perceived as more random.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter for sensitivity to diagnosticity
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Log probability under a fair coin (theta = 0.5)
    log_p_fair_a = n_a * pt.log(0.5)
    log_p_fair_b = n_b * pt.log(0.5)

    # Log probability under a biased coin (uniform prior over theta):
    # This evaluates the Beta-binomial evidence integral:
    # int_0^1 theta^h (1-theta)^(n-h) d(theta) = Gamma(h+1)Gamma(n-h+1)/Gamma(n+2)
    log_p_biased_a = pt.gammaln(h_a + 1) + pt.gammaln(n_a - h_a + 1) - pt.gammaln(n_a + 2)
    log_p_biased_b = pt.gammaln(h_b + 1) + pt.gammaln(n_b - h_b + 1) - pt.gammaln(n_b + 2)

    # Evidence (diagnosticity) for biased coin over fair coin
    diag_a = log_p_biased_a - log_p_fair_a
    diag_b = log_p_biased_b - log_p_fair_b

    # Probability of choosing sequence A (left)
    p_left_raw = pm.math.sigmoid(tau * (diag_a - diag_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
