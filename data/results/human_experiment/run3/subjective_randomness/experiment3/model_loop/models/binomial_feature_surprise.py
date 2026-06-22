"""People judge the randomness of a sequence by intuitively evaluating the exact Binomial probability of its observed number of heads and alternations under a subjective ideal, naturally penalizing deviations more strictly in longer sequences without needing ad-hoc smoothing."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Feature inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))

    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Cognitive parameters
    # The ideal alternation rate people expect
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)

    # The relative weight of alternations vs. heads balance in the log-probability
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    balance_weight = 1.0 - alt_weight

    # Scaling parameter and side bias for the decision rule
    beta = pm.Uniform("beta", lower=0.1, upper=20.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Calculate denominators for alternations
    n_alts_a = pt.maximum(n_a - 1, 1)
    n_alts_b = pt.maximum(n_b - 1, 1)

    # For sequence A
    logp_heads_a = pm.logp(pm.Binomial.dist(n=n_a, p=0.5), h_a)
    logp_alts_a = pm.logp(pm.Binomial.dist(n=n_alts_a, p=theta_alt), alts_a)
    score_a = balance_weight * logp_heads_a + alt_weight * logp_alts_a

    # For sequence B
    logp_heads_b = pm.logp(pm.Binomial.dist(n=n_b, p=0.5), h_b)
    logp_alts_b = pm.logp(pm.Binomial.dist(n=n_alts_b, p=theta_alt), alts_b)
    score_b = balance_weight * logp_heads_b + alt_weight * logp_alts_b

    # Softmax decision rule
    # Sequences with higher log-probability are perceived as MORE random
    p_left_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
