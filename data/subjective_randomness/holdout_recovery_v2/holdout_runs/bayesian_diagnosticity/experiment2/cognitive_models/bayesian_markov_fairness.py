"""People implicitly compute the log-Bayes-factor comparing each sequence's
observed transitions against a fair Markov chain (p_transition = 0.5) versus
a biased one, and choose the sequence whose transitions are more consistent
with the fair-coin hypothesis — a Bayesian inference account rather than a
prototype-heuristic one."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Transition probability of the biased alternative model
    theta_biased = pm.Beta("theta_biased", alpha=2.0, beta=2.0)
    tau = pm.HalfNormal("tau", sigma=2.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    alts_a_f = pt.cast(alts_a, "float64")
    alts_b_f = pt.cast(alts_b, "float64")
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")

    # Number of transitions (n-1 opportunities per sequence)
    steps_a = n_a_f - 1.0
    steps_b = n_b_f - 1.0

    # Clamp theta_biased away from 0/1 for numerical safety
    theta_b = pt.clip(theta_biased, 1e-6, 1.0 - 1e-6)

    # Log-Bayes-factor: log P(alts | fair) - log P(alts | biased)
    # Higher LBF → sequence more consistent with fair coin → judged more random
    log_p_fair_a = steps_a * pt.log(0.5)
    log_p_biased_a = alts_a_f * pt.log(theta_b) + (steps_a - alts_a_f) * pt.log(1.0 - theta_b)
    lbf_a = log_p_fair_a - log_p_biased_a

    log_p_fair_b = steps_b * pt.log(0.5)
    log_p_biased_b = alts_b_f * pt.log(theta_b) + (steps_b - alts_b_f) * pt.log(1.0 - theta_b)
    lbf_b = log_p_fair_b - log_p_biased_b

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (lbf_a - lbf_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
