"""Bayesian randomness judgment with a distorted (over-alternating) model of chance.

A sequence looks random to the extent a fair coin explains it better than a
"regular" generator: a coin with an unknown switch/repeat tendency, or a coin
with an unknown heads bias, each with its unknown rate integrated out under a
uniform prior. The single distortion: people believe a fair coin switches
sides with probability q > 1/2 (fitted), so their "random" likelihood rewards
alternation and penalises repeats. People differ in decision sensitivity.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    # Regular generator 1: Markov coin with unknown switch rate (uniform prior).
    log_markov = math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1)
    # Regular generator 2: biased coin with unknown heads rate (uniform prior).
    log_biased = _log_beta(h + 1, n - h + 1)
    m = max(log_markov, log_biased)
    log_regular = math.log(0.5) + m + math.log(
        math.exp(log_markov - m) + math.exp(log_biased - m)
    )
    return float(k), log_regular


def compute_features(sequence_a, sequence_b):
    k_a, reg_a = _summary(sequence_a)
    k_b, reg_b = _summary(sequence_b)
    return {
        "switch_diff": k_a - k_b,
        "log_regular_diff": reg_a - reg_b,
    }


with pm.Model() as model:
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    log_regular_diff = pm.Data("log_regular_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Believed switch probability of a fair coin (logit scale; 0 = normative).
    logit_q = pm.Normal("logit_q", mu=0.0, sigma=1.0)
    q = pm.Deterministic("q", pm.math.sigmoid(logit_q))

    # Person-specific decision sensitivity (non-centred log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    # Log posterior odds of "random" for left minus right. The random
    # likelihood is 1/2 * q^k * (1-q)^(n-1-k); with equal lengths its
    # difference is switch_diff * logit(q).
    score_diff = switch_diff * logit_q - log_regular_diff

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta[participant_id] * score_diff)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
