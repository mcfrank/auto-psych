"""Extended best model: adds max-run as a fourth non-randomness cue.

iter0_candidate0 combines asymmetric alternation penalty, periodicity, and
imbalance (Dirichlet-weighted, ELPD -484.21). However, max run length is a
qualitatively distinct local cue: a single very long run is perceptually
salient even when the overall alternation rate is moderate. p_alts captures
global streakiness; max_run_norm captures the worst local violation. This
model extends the Dirichlet weight vector from 3 to 4 components, testing
whether max_run_norm provides independent explanatory power.
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
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Prototype alternation rate that feels most random.
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)

    # Asymmetric penalty: streakiness (below prototype) penalized more than
    # over-alternation. streak_k = 1 recovers symmetric penalty.
    streak_k = pm.HalfNormal("streak_k", sigma=2.0)

    # Four feature weights summing to 1: [alt, periodicity, max_run, balance].
    weights = pm.Dirichlet("weights", a=np.ones(4))
    alt_weight = weights[0]
    periodicity_weight = weights[1]
    max_run_weight = weights[2]
    balance_weight = weights[3]

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
        + max_run_weight * max_run_norm_a
        + balance_weight * imbalance_a
    )
    score_b = -(
        alt_weight * asymmetric_alt_penalty(p_alts_b)
        + periodicity_weight * periodicity_b
        + max_run_weight * max_run_norm_b
        + balance_weight * imbalance_b
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
