"""Bayesian generator comparison with a personally distorted fair coin.

People judge randomness as Bayesian model comparison: a sequence looks random
to the extent it is better explained by a fair coin than by the suspected
non-random generators — a biased coin of unknown bias (explains unbalanced H/T
counts) or a Markov coin of unknown switch probability (explains long runs or
rigid alternation), each with a uniform prior, mixed equally. The one
distortion: each person's "fair coin" switches at their own personal rate q_i
instead of 0.5 (hierarchical, non-centred logit-normal), so people disagree
about alternation while still penalising imbalance explained by the biased coin.
"""

import math

import numpy as np
import pymc as pm

MAX_PARTICIPANTS = 400


def _stats(seq):
    seq = seq.strip().upper()
    n = len(seq)
    if n < 2 or set(seq) - {"H", "T"}:
        raise ValueError(f"bad sequence: {seq!r}")
    h = seq.count("H")
    t = n - h
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    # log marginal likelihood under a biased coin, bias ~ Uniform(0, 1)
    log_bias = math.lgamma(h + 1) + math.lgamma(t + 1) - math.lgamma(n + 2)
    # log marginal likelihood under a Markov coin, switch prob ~ Uniform(0, 1)
    log_markov = math.log(0.5) - math.log(n) - math.log(math.comb(n - 1, k))
    m = max(log_bias, log_markov)
    log_alt = math.log(0.5) + m + math.log(math.exp(log_bias - m) + math.exp(log_markov - m))
    return k, n - 1 - k, log_alt


def compute_features(sequence_a, sequence_b):
    ka, ra, la = _stats(sequence_a)
    kb, rb, lb = _stats(sequence_b)
    return {
        "d_switch": float(ka - kb),
        "d_repeat": float(ra - rb),
        "d_log_alt": float(la - lb),
    }


with pm.Model() as model:
    d_switch = pm.Data("d_switch", np.zeros(1, dtype="float64"))
    d_repeat = pm.Data("d_repeat", np.zeros(1, dtype="float64"))
    d_log_alt = pm.Data("d_log_alt", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal believed switch rate of a fair coin (logit scale).
    mu_q = pm.Normal("mu_q", mu=0.4, sigma=1.0)
    sigma_q = pm.HalfNormal("sigma_q", sigma=1.0)
    z_q = pm.Normal("z_q", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    q = pm.Deterministic("q", pm.math.sigmoid(mu_q + sigma_q * z_q))
    # Decision sensitivity to the log posterior-odds difference.
    beta = pm.LogNormal("beta", mu=0.0, sigma=1.0)

    qi = q[participant_id]
    # log P(a|fair_q) - log P(b|fair_q) minus the alternatives' difference
    d_llr = d_switch * pm.math.log(qi) + d_repeat * pm.math.log(1.0 - qi) - d_log_alt
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * d_llr))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
