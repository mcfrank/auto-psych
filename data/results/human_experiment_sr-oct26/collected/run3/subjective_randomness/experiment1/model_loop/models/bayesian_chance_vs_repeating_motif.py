"""Bayesian randomness judgment with an over-alternating model of chance and a
repeating-motif generator among the regular alternatives.

Refinement of `bayesian_overalternating_chance_model`: a sequence looks random
to the extent a fair coin (believed to switch with probability q > 1/2)
explains it better than the regular generators — a Markov coin with unknown
switch rate, a biased coin with unknown heads rate, and (the single change) a
repeating-pattern generator that draws a motif of 1-4 flips uniformly and
repeats it. The repeating-pattern generator receives a fitted share w of the
prior on regularity, so exactly periodic sequences (perfect alternation,
HHTHHT..., HHTTHHTT...) are recognised as patterns and look less random.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NOT_PERIODIC = -60.0


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    log_markov = math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1)
    log_biased = _log_beta(h + 1, n - h + 1)
    m = max(log_markov, log_biased)
    log_regular = math.log(0.5) + m + math.log(
        math.exp(log_markov - m) + math.exp(log_biased - m)
    )
    # Repeating-pattern generator: period p uniform over 1..min(4, n//2)
    # (at least two full repeats), motif uniform over 2^p strings.
    periods = [p for p in range(1, 5) if p <= n // 2]
    prob = 0.0
    for p in periods:
        if all(seq[i] == seq[i - p] for i in range(p, n)):
            prob += (1.0 / len(periods)) * 2.0 ** (-p)
    log_periodic = math.log(prob) if prob > 0 else NOT_PERIODIC
    return float(k), log_regular, log_periodic


def compute_features(sequence_a, sequence_b):
    k_a, reg_a, per_a = _summary(sequence_a)
    k_b, reg_b, per_b = _summary(sequence_b)
    return {
        "switch_diff": k_a - k_b,
        "log_regular_a": reg_a,
        "log_regular_b": reg_b,
        "log_periodic_a": per_a,
        "log_periodic_b": per_b,
    }


with pm.Model() as model:
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    log_regular_a = pm.Data("log_regular_a", np.zeros(1, dtype="float64"))
    log_regular_b = pm.Data("log_regular_b", np.zeros(1, dtype="float64"))
    log_periodic_a = pm.Data("log_periodic_a", np.zeros(1, dtype="float64"))
    log_periodic_b = pm.Data("log_periodic_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Believed switch probability of a fair coin (logit scale; 0 = normative).
    logit_q = pm.Normal("logit_q", mu=0.0, sigma=1.0)
    q = pm.Deterministic("q", pm.math.sigmoid(logit_q))

    # Share of the regularity prior given to the repeating-pattern generator.
    logit_w = pm.Normal("logit_w", mu=-2.0, sigma=1.5)
    w = pm.Deterministic("w", pm.math.sigmoid(logit_w))
    log_w = pt.log(w)
    log_1mw = pt.log1p(-w)

    reg_a = pt.logaddexp(log_1mw + log_regular_a, log_w + log_periodic_a)
    reg_b = pt.logaddexp(log_1mw + log_regular_b, log_w + log_periodic_b)

    # Person-specific decision sensitivity (non-centred log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    score_diff = switch_diff * logit_q - (reg_a - reg_b)

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta[participant_id] * score_diff)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
