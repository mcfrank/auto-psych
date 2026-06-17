"""Prototype-similarity with a threshold on maximum run length.

Existing models penalize runs continuously (max_run_norm proportional to run
length). This model uses a sigmoid threshold instead: runs below a critical
length are largely ignored, while runs at or above the threshold trigger a
discrete-feeling non-randomness signal.

Psychologically this captures a rule-like heuristic — "a run of 4+ is
suspicious" — rather than a smooth distance-from-prototype. The threshold and
its steepness are both inferred from data.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Prototype alternation rate
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)

    # Threshold model for runs: run_threshold is the critical run length at
    # which the sigmoid crosses 0.5; run_sharpness controls how step-like it is.
    run_threshold = pm.Uniform("run_threshold", lower=1.5, upper=6.5)
    run_sharpness = pm.HalfNormal("run_sharpness", sigma=3.0)

    # Feature weights: run_weight allocates mass to the run-threshold term;
    # alt_share splits the remainder between alternation and balance.
    run_weight = pm.Uniform("run_weight", lower=0.01, upper=0.99)
    alt_share = pm.Uniform("alt_share", lower=0.01, upper=0.99)

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = (1.0 - run_weight) * (1.0 - alt_share)
    alt_weight = (1.0 - run_weight) * alt_share

    max_run_a_f = pt.cast(max_run_a, "float64")
    max_run_b_f = pt.cast(max_run_b, "float64")

    # Sigmoid over raw run length: 0 ≈ run is fine, 1 ≈ run is suspicious
    run_penalty_a = pm.math.sigmoid(run_sharpness * (max_run_a_f - run_threshold))
    run_penalty_b = pm.math.sigmoid(run_sharpness * (max_run_b_f - run_threshold))

    def score(imbalance, p_alts, run_penalty):
        return -(
            balance_weight * imbalance
            + alt_weight * pt.abs(p_alts - theta_alt)
            + run_weight * run_penalty
        )

    score_a = score(imbalance_a, p_alts_a, run_penalty_a)
    score_b = score(imbalance_b, p_alts_b, run_penalty_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
