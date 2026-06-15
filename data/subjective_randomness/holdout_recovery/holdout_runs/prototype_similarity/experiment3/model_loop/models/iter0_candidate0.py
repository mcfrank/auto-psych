"""Combined asymmetric-alternation + periodicity model.

The two top models from the current landscape are nearly tied in ELPD:
asymmetric_alternation_prototype (-484.40) uses an asymmetric penalty on p_alts
(streakiness penalised more than over-alternation), and periodicity_salience
(-484.44) uses template-matching periodicity. This model tests whether both
signals contribute independently to randomness judgments, along with imbalance.

A Dirichlet prior over the three feature weights enforces they sum to 1 and
keeps the model identifiable when beta scales the overall decision sharpness.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Ideal alternation rate prototype (what feels most random).
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)

    # Asymmetry: how much more streakiness (below theta_alt) is penalised
    # relative to over-alternation. streak_k = 1 recovers symmetric penalty.
    streak_k = pm.HalfNormal("streak_k", sigma=2.0)

    # Three feature weights summing to 1: [alt_weight, periodicity_weight, balance_weight].
    weights = pm.Dirichlet("weights", a=np.ones(3))
    alt_weight = weights[0]
    periodicity_weight = weights[1]
    balance_weight = weights[2]

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    def asymmetric_alt_penalty(p_alts):
        return (
            streak_k * pt.maximum(theta_alt - p_alts, 0.0)
            + pt.maximum(p_alts - theta_alt, 0.0)
        )

    score_a = -(
        alt_weight * asymmetric_alt_penalty(p_alts_a)
        + periodicity_weight * periodicity_a
        + balance_weight * imbalance_a
    )
    score_b = -(
        alt_weight * asymmetric_alt_penalty(p_alts_b)
        + periodicity_weight * periodicity_b
        + balance_weight * imbalance_b
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
