"""
People judge a coin sequence as more random-looking when it falls close to
an internal two-dimensional prototype specifying both an ideal alternation
rate and balanced heads-to-tails proportions. Their sensitivity to departures
is linear (L1 / absolute-value distance), not quadratic: each unit of
deviation from the ideal reduces perceived randomness by a constant amount
regardless of how far from the prototype the sequence already is.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_a      = pm.Data("p_a",      np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    p_b      = pm.Data("p_b",      np.zeros(1, dtype="float64"))

    # Free cognitive parameters.
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    beta      = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # L1 distance from the 2D prototype — linear, not quadratic, decay.
    score_a = -pt.abs(p_alts_a - theta_alt) - pt.abs(p_a - 0.5)
    score_b = -pt.abs(p_alts_b - theta_alt) - pt.abs(p_b - 0.5)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
