"""Asymmetric prototype model: streakiness is penalized more than over-alternation.

The symmetric |p_alts - theta_alt| term in the best experiment-1 model treats
being too streaky and being too alternating as equally non-random. Gambler's
fallacy research shows people react more strongly to runs (streakiness) than to
excess alternation. This model introduces streak_k >= 0 to scale the below-prototype
penalty separately. When streak_k = 1 the model reduces exactly to the symmetric
prototype model."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    # streak_k > 1: streakiness penalized more than over-alternation;
    # streak_k = 1 recovers symmetric |p_alts - theta_alt|.
    streak_k = pm.HalfNormal("streak_k", sigma=2.0)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight

    def asymmetric_alt_penalty(p_alts):
        return (
            streak_k * pt.maximum(theta_alt - p_alts, 0.0)
            + pt.maximum(p_alts - theta_alt, 0.0)
        )

    score_a = -(balance_weight * imbalance_a + alt_weight * asymmetric_alt_penalty(p_alts_a))
    score_b = -(balance_weight * imbalance_b + alt_weight * asymmetric_alt_penalty(p_alts_b))

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
