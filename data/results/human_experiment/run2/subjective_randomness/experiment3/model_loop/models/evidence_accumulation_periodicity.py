"""PyMC model for the evidence accumulation with periodicity penalty hypothesis.

Random-looking sequences are judged by an evidence accumulation process where baseline evidence is discounted by the sequence's deviation from a messy prototype, but rather than tracking local alternation rates, people explicitly penalize global repeating templates (periodicity). The prototype thus consists of an ideal imbalance and an ideal (low) periodicity, naturally explaining the harsh rejection of structured, repeating sequences that a simple alternation-rate penalty misses.
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
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    theta_per = pm.Uniform("theta_per", lower=0.0, upper=0.5)
    theta_imb = pm.Uniform("theta_imb", lower=0.0, upper=1.0)
    per_weight = pm.Uniform("per_weight", lower=0.01, upper=0.99)

    # Baseline evidence per item
    base_evidence = pm.HalfNormal("base_evidence", sigma=2.0)

    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    balance_weight = 1.0 - per_weight

    # Quadratic distance for periodicity
    per_penalty_a = (periodicity_a - theta_per) ** 2
    per_penalty_b = (periodicity_b - theta_per) ** 2

    # Quadratic distance for messy imbalance prototype
    imb_penalty_a = (imbalance_a - theta_imb) ** 2
    imb_penalty_b = (imbalance_b - theta_imb) ** 2

    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")

    # Evidence accumulation: length * (baseline - penalty)
    score_a = n_a_f * (
        base_evidence - (balance_weight * imb_penalty_a + per_weight * per_penalty_a)
    )
    score_b = n_b_f * (
        base_evidence - (balance_weight * imb_penalty_b + per_weight * per_penalty_b)
    )

    # Sigmoid link to probability, clamped for numerical safety
    p_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_raw, 1e-6, 1.0 - 1e-6))

    # Likelihood
    pm.Bernoulli("response", p=p_left, observed=chose_left)
