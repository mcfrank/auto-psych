"""Bayesian rigged-coin detection with a two-flip-memory rigged coin.

Refinement of `iter0_candidate0`: a sequence looks random to the extent it is
better explained by each person's subjective random coin (switching at their own
personal rate) than by a rigged process. The rigged processes are a coin biased
toward one face, a sticky/switchy coin, and -- the one change -- a coin with a
two-flip memory (an unknown next-flip probability for each of the four two-flip
contexts, uniform priors), all equally likely a priori. Streaks and repeating
patterns reuse few triplets, are well explained by the memory coin, and so look
rigged even at equal switch counts.
"""
import math

import numpy as np
import pymc as pm


def _log_beta_marginal(k, n):
    # log of integral p^k (1-p)^(n-k) dp under a uniform prior.
    return math.lgamma(k + 1) + math.lgamma(n - k + 1) - math.lgamma(n + 2)


def _log_p_rigged(seq):
    n = len(seq)
    heads = seq.count("H")
    log_biased = _log_beta_marginal(heads, n)
    switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    log_markov = math.log(0.5) + _log_beta_marginal(switches, n - 1)
    # Two-flip memory coin: first two flips fair, then per-context Beta(1,1).
    counts = {}
    for i in range(2, n):
        ctx = seq[i - 2:i]
        h, t = counts.get(ctx, (0, 0))
        counts[ctx] = (h + (seq[i] == "H"), t + (seq[i] != "H"))
    log_memory = min(n, 2) * math.log(0.5)
    for h, t in counts.values():
        log_memory += _log_beta_marginal(h, h + t)
    terms = [log_biased, log_markov, log_memory]
    m = max(terms)
    return m + math.log(sum(math.exp(x - m) for x in terms) / 3.0)


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()

    def switches(seq):
        return float(sum(1 for x, y in zip(seq, seq[1:]) if x != y))

    return {
        # Equal lengths: the subjective random coin's log-likelihood difference
        # is (switches_a - switches_b) * logit(personal switch rate).
        "switch_diff": switches(a) - switches(b),
        # log P(b | rigged) - log P(a | rigged): evidence against a's rigging.
        "rigged3_diff": _log_p_rigged(b) - _log_p_rigged(a),
    }


N_SLOTS = 400

with pm.Model() as model:
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    rigged_diff = pm.Data("rigged3_diff", np.zeros(1, dtype="float64"))
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
