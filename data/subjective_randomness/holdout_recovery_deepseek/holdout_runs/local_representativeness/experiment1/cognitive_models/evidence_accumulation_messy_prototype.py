"""PyMC model for the evidence accumulation messy-prototype hypothesis.

Random-looking sequences are judged by an evidence accumulation process where each item in the sequence provides a baseline "weight of evidence" for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype (an ideal positive imbalance and ideal alternation rate). This mechanism naturally creates a preference for longer sequences when they are near-ideal, but more heavily penalizes long sequences that clearly deviate from the prototype.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    theta_imb = pm.Uniform("theta_imb", lower=0.0, upper=1.0)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    # Asymmetry parameter: < 0.5 means over-alternation is penalized less than under-alternation
    asym = pm.Uniform("asym", lower=0.01, upper=0.99)

    # Baseline evidence per item
    base_evidence = pm.HalfNormal("base_evidence", sigma=2.0)

    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    balance_weight = 1.0 - alt_weight

    # Quadratic asymmetric distance for alternation rate
    diff_alt_a = p_alts_a - theta_alt
    alt_penalty_a = pm.math.switch(
        diff_alt_a > 0,
        2.0 * asym * (diff_alt_a**2),
        2.0 * (1.0 - asym) * (diff_alt_a**2),
    )

    diff_alt_b = p_alts_b - theta_alt
    alt_penalty_b = pm.math.switch(
        diff_alt_b > 0,
        2.0 * asym * (diff_alt_b**2),
        2.0 * (1.0 - asym) * (diff_alt_b**2),
    )

    # Quadratic distance for messy imbalance prototype
    imb_penalty_a = (imbalance_a - theta_imb) ** 2
    imb_penalty_b = (imbalance_b - theta_imb) ** 2

    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")

    # Evidence accumulation: length * (baseline - penalty)
    score_a = n_a_f * (
        base_evidence - (balance_weight * imb_penalty_a + alt_weight * alt_penalty_a)
    )
    score_b = n_b_f * (
        base_evidence - (balance_weight * imb_penalty_b + alt_weight * alt_penalty_b)
    )

    # Sigmoid link to probability, clamped for numerical safety
    p_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_raw, 1e-6, 1.0 - 1e-6))

    # Likelihood
    pm.Bernoulli("response", p=p_left, observed=chose_left)
