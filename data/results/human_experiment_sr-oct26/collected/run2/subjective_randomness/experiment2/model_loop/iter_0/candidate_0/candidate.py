"""Bayesian detection of a rigged coin, with a personally distorted random coin.

People judge randomness by Bayesian model comparison: a sequence looks random to
the extent it is better explained by a random coin than by a "rigged" process --
either a coin biased toward one face (unknown bias, uniform prior) or a coin
that tends to repeat or alternate (unknown repeat probability, uniform prior),
each equally likely a priori. Lopsided H/T counts and extreme switching both
become evidence for rigging. The single distortion: each person's random coin is
not the fair memoryless coin but one that switches sides at their own personal
rate. They choose the sequence with the higher log posterior odds of randomness.
"""
import math

import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def log_beta_marginal(k, n):
        # log of integral p^k (1-p)^(n-k) dp under a uniform prior.
        return math.lgamma(k + 1) + math.lgamma(n - k + 1) - math.lgamma(n + 2)

    def log_p_rigged(seq):
        n = len(seq)
        heads = seq.count("H")
        log_biased = log_beta_marginal(heads, n)
        t = n - 1
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        log_markov = math.log(0.5) + log_beta_marginal(switches, t)
        m = max(log_biased, log_markov)
        return m + math.log(0.5 * math.exp(log_biased - m) + 0.5 * math.exp(log_markov - m))

    def switches(seq):
        return float(sum(1 for x, y in zip(seq, seq[1:]) if x != y))

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        # Equal lengths: the subjective random coin's log-likelihood difference
        # is (switches_a - switches_b) * logit(personal switch rate).
        "switch_diff": switches(a) - switches(b),
        # log P(b | rigged) - log P(a | rigged): evidence against a's rigging.
        "rigged_diff": log_p_rigged(b) - log_p_rigged(a),
    }


N_SLOTS = 400

with pm.Model() as model:
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    rigged_diff = pm.Data("rigged_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal belief about a random coin's switch rate (logit scale).
    mu_switch = pm.Normal("mu_switch", mu=0.3, sigma=1.0)
    sigma_switch = pm.HalfNormal("sigma_switch", sigma=0.7)
    z_switch = pm.Normal("z_switch", mu=0.0, sigma=1.0, shape=N_SLOTS)
    logit_switch = mu_switch + sigma_switch * z_switch

    # Personal scaling of the log posterior odds into choices.
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z_beta)

    log_odds_diff = switch_diff * logit_switch[participant_id] + rigged_diff
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta[participant_id] * log_odds_diff)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
