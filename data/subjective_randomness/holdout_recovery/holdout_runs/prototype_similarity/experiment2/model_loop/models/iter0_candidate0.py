"""Prototype-similarity model augmented with max-run-length.

The two tied best models (inner_loop_model, asymmetric_alternation_prototype)
both use imbalance + p_alts.  max_run_norm captures the worst-case streak
within a sequence — a distinct cue from the mean alternation rate.  A Dirichlet
prior over the three feature weights lets the data arbitrate how much each
dimension contributes to perceived randomness.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Prototype alternation rate that feels "most random".
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)

    # Feature weights: (balance, alternation, run-length) — sum to 1.
    weights = pm.Dirichlet("weights", a=np.ones(3))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Higher penalty → less random.  max_run_norm is already in [0,1] and
    # larger values indicate longer streaks, so it enters as a direct penalty.
    def prototype_score(imbalance, p_alts, max_run_norm):
        return -(
            weights[0] * imbalance
            + weights[1] * pt.abs(p_alts - theta_alt)
            + weights[2] * max_run_norm
        )

    score_a = prototype_score(imbalance_a, p_alts_a, max_run_norm_a)
    score_b = prototype_score(imbalance_b, p_alts_b, max_run_norm_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
