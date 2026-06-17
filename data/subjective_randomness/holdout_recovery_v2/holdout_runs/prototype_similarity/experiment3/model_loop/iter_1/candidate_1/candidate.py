"""
People judge sequences by Bayesian diagnosticity of a fair coin against three
non-random alternatives (alternating, biased, streaky). The alternating and
biased generator parameters are fixed at canonical values, but the streaky
generator's switch probability is a learned cognitive parameter.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Sensitivity to diagnosticity differences
    tau = pm.HalfNormal("tau", sigma=2.0)
    # Streaky generator switch probability — learned, prior centered near canonical 0.15
    q_streak = pm.Beta("q_streak", alpha=2.0, beta=8.0)
    q_streak_safe = pt.clip(q_streak, 1e-6, 1.0 - 1e-6)

    # Fixed canonical generator parameters
    q_alt = 0.95    # alternating generator: almost always switches
    p_bias = 0.85   # biased generator: strong heads preference

    # Log P(sequence | fair coin)
    log_fair_a = n_a * np.log(0.5)
    log_fair_b = n_b * np.log(0.5)

    # Log P(sequence | alternating generator) — uses transition counts
    log_alt_a = alts_a * np.log(q_alt) + (n_a - 1 - alts_a) * np.log(1.0 - q_alt)
    log_alt_b = alts_b * np.log(q_alt) + (n_b - 1 - alts_b) * np.log(1.0 - q_alt)

    # Log P(sequence | biased generator) — uses head counts
    log_biased_a = h_a * np.log(p_bias) + (n_a - h_a) * np.log(1.0 - p_bias)
    log_biased_b = h_b * np.log(p_bias) + (n_b - h_b) * np.log(1.0 - p_bias)

    # Log P(sequence | streaky generator) — uses transition counts, learned q
    log_streak_a = (
        alts_a * pt.log(q_streak_safe)
        + (n_a - 1 - alts_a) * pt.log(1.0 - q_streak_safe)
    )
    log_streak_b = (
        alts_b * pt.log(q_streak_safe)
        + (n_b - 1 - alts_b) * pt.log(1.0 - q_streak_safe)
    )

    # Diagnosticity = log P(fair) minus log P under most compelling non-random alternative
    diag_a = log_fair_a - pt.maximum(pt.maximum(log_alt_a, log_biased_a), log_streak_a)
    diag_b = log_fair_b - pt.maximum(pt.maximum(log_alt_b, log_biased_b), log_streak_b)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (diag_a - diag_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
