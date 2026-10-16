"""Paired comparison with mismatch-scaled noise.

People compare the two sequences flip against flip: positions where both show
the same outcome cancel, and only mismatching positions carry evidence and
noise into the judgment. The randomness evidence of each sequence is the
Bayesian "fair coin (believed to over-alternate) vs regular generator" score;
the comparison noise grows with the number of positions at which the two
sequences differ (Thurstonian comparison with shared, cancelling components),
so the same score difference is judged more decisively beside a similar
partner than beside a very different one.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    log_markov = math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1)
    log_biased = _log_beta(h + 1, n - h + 1)
    m = max(log_markov, log_biased)
    log_regular = math.log(0.5) + m + math.log(
        math.exp(log_markov - m) + math.exp(log_biased - m)
    )
    return float(k), log_regular


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    k_a, reg_a = _summary(a)
    k_b, reg_b = _summary(b)
    mismatches = sum(1 for x, y in zip(a, b) if x != y)
    return {
        "switch_diff": k_a - k_b,
        "log_regular_diff": reg_a - reg_b,
        "n_mismatch": float(mismatches),
    }


with pm.Model() as model:
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    log_regular_diff = pm.Data("log_regular_diff", np.zeros(1, dtype="float64"))
    n_mismatch = pm.Data("n_mismatch", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Believed switch probability of a fair coin (logit scale).
    logit_q = pm.Normal("logit_q", mu=0.0, sigma=1.0)

    # Comparison noise variance added per mismatching position.
    gamma = pm.HalfNormal("gamma", sigma=1.0)

    # Person-specific decision sensitivity (non-centred log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    score_diff = switch_diff * logit_q - log_regular_diff
    noise_sd = pt.sqrt(1.0 + gamma * n_mismatch)

    p_left = pm.Deterministic(
        "p_left",
        pt.clip(
            pm.math.sigmoid(beta[participant_id] * score_diff / noise_sd),
            1e-6,
            1 - 1e-6,
        ),
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
