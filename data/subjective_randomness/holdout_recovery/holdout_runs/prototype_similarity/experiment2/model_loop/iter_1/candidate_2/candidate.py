"""Hierarchical prototype model: participant-level variation in the random prototype.

All existing models use a single population-level prototype alternation rate
(theta_alt). Different people may have different intuitions about what alternation
frequency "looks random." This model places a hierarchical prior over
per-participant theta_alt values drawn from a population distribution.

With 30 participants x 40 trials, there is enough data to partially identify
individual prototypes. The non-centered parameterization ensures efficient NUTS
sampling. If participants vary substantially in their prototype, pooling that
variation into a single theta_alt inflates residual variance and hurts ELPD.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_PARTICIPANTS = 30

with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Population mean prototype alternation rate (logit scale — unconstrained).
    mu_theta_logit = pm.Normal("mu_theta_logit", mu=0.0, sigma=1.0)
    # Between-participant spread in prototype.
    sigma_theta = pm.HalfNormal("sigma_theta", sigma=0.5)

    # Non-centered per-participant offsets for efficient NUTS.
    theta_offset = pm.Normal("theta_offset", mu=0, sigma=1, shape=N_PARTICIPANTS)

    # Per-participant prototype rate in (0, 1).
    theta_alt_all = pm.Deterministic(
        "theta_alt_all",
        pm.math.sigmoid(mu_theta_logit + sigma_theta * theta_offset),
    )

    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight

    # Index to the participant-specific prototype for each trial.
    theta_p = theta_alt_all[participant_id]

    score_a = -(balance_weight * imbalance_a + alt_weight * pt.abs(p_alts_a - theta_p))
    score_b = -(balance_weight * imbalance_b + alt_weight * pt.abs(p_alts_b - theta_p))

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
