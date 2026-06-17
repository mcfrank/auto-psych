"""Lapse-rate prototype model: mixture of attentive and random responding.

All models in the current landscape assume every response reflects the same
deliberate judgment. This model relaxes that assumption: with probability
epsilon (lapse rate) the participant responds at random (50/50), and with
probability 1-epsilon they compare sequences by squared distance from an
ideal alternation-rate prototype.

The core signal is a symmetric quadratic penalty |(p_alts - theta_alt)^2|,
simpler than the asymmetric linear penalty in the top-ranked model. The lapse
mixture is structurally different from all existing models and tests whether
response noise is responsible for the residual unexplained variance that keeps
every model near ELPD -484.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Ideal alternation rate that feels most random.
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)

    # Decision sensitivity; HalfNormal regularizes toward lower sharpness.
    beta = pm.HalfNormal("beta", sigma=4.0)

    # Lapse rate: probability of random (50/50) response on any trial.
    epsilon = pm.Beta("epsilon", alpha=1.0, beta=9.0)

    # Left-right response bias.
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Squared deviation from prototype: larger = more non-random.
    penalty_a = (p_alts_a - theta_alt) ** 2
    penalty_b = (p_alts_b - theta_alt) ** 2

    # Attentive response probability.
    p_attentive = pm.math.sigmoid(beta * (-penalty_a - (-penalty_b)) + side_bias)

    # Mixture: lapse trials pull toward 0.5.
    p_left = pm.Deterministic("p_left", (1.0 - epsilon) * p_attentive + epsilon * 0.5)

    pm.Bernoulli("response", p=p_left, observed=chose_left)
