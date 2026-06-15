"""Alternation z-score model: non-randomness measured relative to IID expectation.

All prototype models compare p_alts to a fixed or learned reference theta_alt.
This model replaces that prototype distance with a normalized z-score:

    z = (p_alts - mu_IID) / sqrt(Var_IID)

where mu_IID = 2p(1-p) and Var_IID = mu_IID*(1-mu_IID)/(n-1) are the IID-expected
alternation rate and its variance for a sequence of length n with empirical bias p =
h/n. The z-score is signed: negative means fewer alternations than IID predicts
(streaky); positive means more (over-alternating).

Key differences from existing models:
- No learned theta_alt: the IID-expected rate is computed directly from each
  sequence's own balance and length.
- Automatic length calibration: a 2-SD deviation in n=8 and n=20 sequences carries
  equal cognitive weight, unlike unnormalized deviations.
- Handles variable coin biases correctly: for an all-heads sequence from a p=1.0
  coin, z=0 (no alternation deviation) while imbalance still flags the balance cue.

Periodicity and imbalance are retained as secondary cues; the Dirichlet prior over
the three feature weights allows the model to discover which cues matter most.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Asymmetric sensitivity: streakiness (negative z) penalised more than
    # over-alternation (positive z), consistent with the gambler's-fallacy direction.
    streak_k = pm.HalfNormal("streak_k", sigma=2.0)

    # Three feature weights summing to 1: [alt_zscore, periodicity, imbalance].
    weights = pm.Dirichlet("weights", a=np.ones(3))
    alt_weight = weights[0]
    periodicity_weight = weights[1]
    balance_weight = weights[2]

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    def z_score_penalty(n, h, p_alts):
        n_f = pt.cast(n, "float64")
        h_f = pt.cast(h, "float64")
        p = h_f / pt.maximum(n_f, 1.0)
        # IID-predicted alternation rate and its variance for this sequence.
        mu_alts = 2.0 * p * (1.0 - p)
        var_alts = mu_alts * (1.0 - mu_alts) / pt.maximum(n_f - 1.0, 1.0)
        z = (p_alts - mu_alts) / pt.sqrt(pt.maximum(var_alts, 1e-8))
        return streak_k * pt.maximum(-z, 0.0) + pt.maximum(z, 0.0)

    score_a = -(
        alt_weight * z_score_penalty(n_a, h_a, p_alts_a)
        + periodicity_weight * periodicity_a
        + balance_weight * imbalance_a
    )
    score_b = -(
        alt_weight * z_score_penalty(n_b, h_b, p_alts_b)
        + periodicity_weight * periodicity_b
        + balance_weight * imbalance_b
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
