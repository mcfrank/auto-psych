"""Lapse-rate prototype model.

The top two models (inner_loop_model, asymmetric_alternation_prototype) both
use imbalance + alternation-rate prototype similarity. Candidates 0-1 explored
adding run-length features. This model instead adds a lapse rate parameter:
on fraction epsilon of trials the participant responds at chance (0.5),
and on the remaining 1-epsilon they use the prototype distance signal.

Psychologically this captures attention lapses or guessing, which is common
in forced-choice tasks. It is one parameter added to the simplest top model
and addresses a different explanatory dimension than run length.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Prototype alternation rate: what alternation frequency feels "random"?
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Lapse rate: probability of responding randomly regardless of stimulus.
    # Beta(1,9) prior: most mass near 0, allows up to ~30% lapses.
    epsilon = pm.Beta("epsilon", alpha=1.0, beta=9.0)

    balance_weight = 1.0 - alt_weight
    score_a = -(balance_weight * imbalance_a + alt_weight * pt.abs(p_alts_a - theta_alt))
    score_b = -(balance_weight * imbalance_b + alt_weight * pt.abs(p_alts_b - theta_alt))

    p_signal = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)

    # Mix attentive responses with lapse-rate chance responses.
    p_left = pm.Deterministic("p_left", (1.0 - epsilon) * p_signal + epsilon * 0.5)

    pm.Bernoulli("response", p=p_left, observed=chose_left)
