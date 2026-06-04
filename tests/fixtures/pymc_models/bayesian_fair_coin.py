# file: bayesian_fair_coin.py
"""Observers compare two binary sequences via log Bayes factor between a
fair-coin null and a biased-coin alternative, then pick the more fair-coin-like
sequence with a softmax decision rule."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    theta = pm.Beta("theta", alpha=2.0, beta=2.0)
    tau = pm.HalfNormal("tau", sigma=2.0)

    log_fair_a = pt.cast(n_a, "float64") * pt.log(0.5)
    log_bias_a = pt.cast(h_a, "float64") * pt.log(theta) + pt.cast(n_a - h_a, "float64") * pt.log(1.0 - theta)
    lbf_a = log_fair_a - log_bias_a

    log_fair_b = pt.cast(n_b, "float64") * pt.log(0.5)
    log_bias_b = pt.cast(h_b, "float64") * pt.log(theta) + pt.cast(n_b - h_b, "float64") * pt.log(1.0 - theta)
    lbf_b = log_fair_b - log_bias_b

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (lbf_a - lbf_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
