"""Minimal symmetric alternation prototype model.

The dominant models in this landscape (iter0_candidate0 and asymmetric_alternation
prototype) use 6–7 parameters: a learned theta_alt, streak asymmetry (streak_k),
a Dirichlet over 3 feature weights, beta, and side_bias. This model tests whether
all that complexity is necessary by stripping back to the single most important
cognitive principle: people have a learned alternation-rate prototype and choose
whichever sequence is closer to it.

Differences from the top models:
- No asymmetry (streak_k removed): equal penalty for streakiness and over-alternation.
- No secondary features: periodicity and imbalance are dropped entirely.
- 3 free parameters total: theta_alt, beta, side_bias.

If this performs near the top models it implies the asymmetry and secondary features
explain very little variance. If it performs substantially worse it quantifies how
much those features and asymmetry contribute.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Learned prototype: alternation rate that feels most random.
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)

    # Decision sharpness.
    beta = pm.HalfNormal("beta", sigma=4.0)

    # Response side bias.
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Randomness score: symmetric distance from prototype (closer = more random).
    score_a = -pt.abs(p_alts_a - theta_alt)
    score_b = -pt.abs(p_alts_b - theta_alt)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
