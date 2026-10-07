"""Leaky-memory Bayesian judgment of chance versus regular generators.

People judge randomness as Bayesian inference: their own model of a fair coin
(a person-specific believed switch rate) against regular generators (a
switch-biased Markov coin, a biased coin, a short motif repeated with slips),
with the generators' unknowns averaged out. The single distortion is a leaky
memory: they read the sequence flip by flip and each flip's evidence (its
predictive log probability under chance minus under the regular mixture) is
weighted by exp(-rho * lag), lag = number of flips read after it. rho > 0:
early evidence fades (recency); rho < 0: first impressions dominate (primacy);
rho = 0 recovers the normative total evidence.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)
EPS = 0.12  # slip rate of the repeating-motif generator
MAXLAG = 8


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _log_regular_marginal(seq):
    """Log marginal probability of a prefix under the equal-weight regular mixture."""
    n = len(seq)
    if n == 0:
        return 0.0
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    lm = math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1)
    lb = _log_beta(h + 1, n - h + 1)
    motif = []
    for p in PERIODS:
        free = min(p, n)
        cmp_ = max(n - p, 0)
        mis = sum(1 for i in range(p, n) if seq[i] != seq[i - p])
        motif.append(-free * math.log(2.0) + (cmp_ - mis) * math.log(1 - EPS) + mis * math.log(EPS))
    m = max(motif)
    lmo = m + math.log(sum(math.exp(v - m) for v in motif) / len(PERIODS))
    terms = [lm, lb, lmo]
    m = max(terms)
    return m + math.log(sum(math.exp(v - m) for v in terms) / 3.0)


def _per_lag(seq):
    """Per-flip switch indicators and regular predictive log probs, indexed by lag from the end."""
    seq = seq.strip().upper()
    n = len(seq)
    s = [0.0] * MAXLAG
    r = [0.0] * MAXLAG
    prev = 0.0
    for t in range(n):
        cur = _log_regular_marginal(seq[: t + 1])
        lag = n - 1 - t
        r[lag] = cur - prev
        prev = cur
        if t >= 1 and seq[t] != seq[t - 1]:
            s[lag] = 1.0
    return s, r, n


def compute_features(sequence_a, sequence_b):
    sa, ra, n = _per_lag(sequence_a)
    sb, rb, _ = _per_lag(sequence_b)
    out = {"seq_n": float(n)}
    for k in range(MAXLAG):
        out[f"sd{k}"] = sa[k] - sb[k]
        out[f"rd{k}"] = ra[k] - rb[k]
    return out


with pm.Model() as model:
    seq_n = pm.Data("seq_n", np.full(1, 2.0))
    sd = pt.stack([pm.Data(f"sd{k}", np.zeros(1, dtype="float64")) for k in range(MAXLAG)], axis=1)
    rd = pt.stack([pm.Data(f"rd{k}", np.zeros(1, dtype="float64")) for k in range(MAXLAG)], axis=1)
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Memory leak per flip read afterwards (log scale of the decay).
    rho = pm.Normal("rho", mu=0.0, sigma=0.5)
    lags = pt.arange(MAXLAG, dtype="float64")
    wts = pt.exp(-rho * lags)

    switch_ev = pt.dot(sd, wts)  # weighted switch-count difference
    regular_ev = pt.dot(rd, wts)  # weighted regular log-predictive difference

    # Person-specific decision sensitivity.
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    # Person-specific believed switch log-odds times sensitivity (non-centred).
    mu_w = pm.Normal("mu_w", mu=0.0, sigma=1.0)
    sigma_w = pm.HalfNormal("sigma_w", sigma=0.5)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=400)
    w = mu_w + sigma_w * z_w

    # Length normalisation of the evidence.
    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    length_scale = pt.exp(-gamma * pt.log(seq_n / 5.0))

    pid = participant_id
    eta = (w[pid] * switch_ev - beta[pid] * regular_ev) * length_scale
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
