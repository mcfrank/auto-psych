"""Random-looking sequences are judged by an evidence accumulation process where each item provides a baseline weight of evidence for randomness, which is then discounted by the sequence's quadratic deviation from a messy prototype defined by an ideal repeating-motif rate and an ideal alternating-motif rate."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    rep_motifs_a = pm.Data("rep_motifs_a", np.zeros(1, dtype="int64"))
    rep_motifs_b = pm.Data("rep_motifs_b", np.zeros(1, dtype="int64"))
    alt_motifs_a = pm.Data("alt_motifs_a", np.zeros(1, dtype="int64"))
    alt_motifs_b = pm.Data("alt_motifs_b", np.zeros(1, dtype="int64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    theta_rep = pm.Uniform("theta_rep", lower=0.0, upper=1.0)
    theta_alt = pm.Uniform("theta_alt", lower=0.0, upper=1.0)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    # Asymmetry parameter for alternating motifs
    asym = pm.Uniform("asym", lower=0.01, upper=0.99)

    # Baseline evidence per item
    base_evidence = pm.HalfNormal("base_evidence", sigma=2.0)

    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    balance_weight = 1.0 - alt_weight

    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")
    
    p_rep_a = pt.cast(rep_motifs_a, "float64") / pt.clip(n_a_f, 1.0, np.inf)
    p_rep_b = pt.cast(rep_motifs_b, "float64") / pt.clip(n_b_f, 1.0, np.inf)
    
    p_alt_a = pt.cast(alt_motifs_a, "float64") / pt.clip(n_a_f, 1.0, np.inf)
    p_alt_b = pt.cast(alt_motifs_b, "float64") / pt.clip(n_b_f, 1.0, np.inf)

    # Quadratic asymmetric distance for alternating motifs
    diff_alt_a = p_alt_a - theta_alt
    alt_penalty_a = pm.math.switch(
        diff_alt_a > 0,
        2.0 * asym * (diff_alt_a**2),
        2.0 * (1.0 - asym) * (diff_alt_a**2),
    )

    diff_alt_b = p_alt_b - theta_alt
    alt_penalty_b = pm.math.switch(
        diff_alt_b > 0,
        2.0 * asym * (diff_alt_b**2),
        2.0 * (1.0 - asym) * (diff_alt_b**2),
    )

    # Quadratic distance for repeating motifs prototype
    rep_penalty_a = (p_rep_a - theta_rep) ** 2
    rep_penalty_b = (p_rep_b - theta_rep) ** 2

    # Evidence accumulation: length * (baseline - penalty)
    score_a = n_a_f * (
        base_evidence - (balance_weight * rep_penalty_a + alt_weight * alt_penalty_a)
    )
    score_b = n_b_f * (
        base_evidence - (balance_weight * rep_penalty_b + alt_weight * alt_penalty_b)
    )

    # Sigmoid link to probability, clamped for numerical safety
    p_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_raw, 1e-6, 1.0 - 1e-6))

    # Likelihood
    pm.Bernoulli("response", p=p_left, observed=chose_left)
