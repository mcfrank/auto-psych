"""Bayesian randomness judgment with a person-specific belief about how often a fair coin switches.

Refinement of `bayesian_chance_vs_motif_beta_trick_coin` (single change): the
believed switch probability of a fair coin, q, is person-specific (logit_q_i =
mu_q + sigma_q * z_i, non-centred), rather than shared. Most people may expect
chance to over-alternate (logit_q_i > 0), but some believe a fair coin tends to
repeat (logit_q_i < 0) and so find streakier sequences more random.

Parent model description:

Bayesian randomness judgment with a fitted belief about how biased a trick coin is.

Refinement of `bayesian_chance_vs_repeating_motif_2` (single change): the
biased-coin regular generator's unknown heads rate has a symmetric Beta(a, a)
prior with a fitted concentration a, instead of a fixed uniform prior. a < 1:
people imagine trick coins as strongly biased (only lopsided sequences are
suspicious); a > 1: they imagine mildly biased coins, so even a modest
heads/tails imbalance counts as evidence of a biased coin.

Original description:

Refinement of `bayesian_overalternating_chance_model`. A sequence looks random
to the extent a fair coin (believed to switch with probability q > 1/2, fitted)
explains it better than a "regular" generator. The regular hypotheses are, as
before, a Markov coin with unknown switch rate and a biased coin with unknown
heads rate (rates integrated out under uniform priors), plus one new one: a
repeating-motif process that picks a period p in {1..4} and a motif of that
length uniformly, then copies the flip p positions back with probability
1 - eps (eps, the slip rate, fitted). Near-periodic sequences (perfect
alternation, period-3/4 patterns) are then explained as regular.
People differ in decision sensitivity.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    log_markov = math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1)
    log_biased = _log_beta(h + 1, n - h + 1)
    out = {"k": float(k), "lm": log_markov, "lb": log_biased, "h": float(h)}
    for p in PERIODS:
        compared = max(n - p, 0)
        mism = sum(1 for i in range(p, n) if seq[i] != seq[i - p])
        out[f"c{p}"] = float(compared)
        out[f"m{p}"] = float(mism)
    out["n"] = float(n)
    return out


def prepare_observed(rows):
    """Table of distinct sequences (regular-generator statistics) plus per-trial indices."""
    index = {}
    table = []
    idx_a, idx_b, switch_diff, pid, y = [], [], [], [], []
    for r in rows:
        ia_ib = []
        for key in ("sequence_a", "sequence_b"):
            seq = str(r[key]).strip().upper()
            if seq not in index:
                index[seq] = len(table)
                table.append(_summary(seq))
            ia_ib.append(index[seq])
        idx_a.append(ia_ib[0])
        idx_b.append(ia_ib[1])
        switch_diff.append(table[ia_ib[0]]["k"] - table[ia_ib[1]]["k"])
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    out = {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "switch_diff": np.asarray(switch_diff, dtype="float64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "seq_lmark": np.asarray([t["lm"] for t in table], dtype="float64"),
        "seq_len": np.asarray([t["n"] for t in table], dtype="float64"),
        "seq_heads": np.asarray([t["h"] for t in table], dtype="float64"),
    }
    for p in PERIODS:
        out[f"seq_cmp{p}"] = np.asarray([t[f"c{p}"] for t in table], dtype="float64")
        out[f"seq_mis{p}"] = np.asarray([t[f"m{p}"] for t in table], dtype="float64")
    return out


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_heads = pm.Data("seq_heads", np.ones(1, dtype="float64"))
    seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}

    # Believed switch probability of a fair coin (logit scale; 0 = normative).
    # Person-specific (non-centred population; spare slots predict new people).
    mu_logit_q = pm.Normal("mu_logit_q", mu=0.0, sigma=1.0)
    sigma_logit_q = pm.HalfNormal("sigma_logit_q", sigma=0.5)
    z_q = pm.Normal("z_q", mu=0.0, sigma=1.0, shape=400)
    logit_q = mu_logit_q + sigma_logit_q * z_q

    # Slip rate of the repeating-motif generator.
    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    eps = pm.Deterministic("eps", pm.math.sigmoid(logit_eps))
    log_eps = pt.log(eps)
    log_1m_eps = pt.log1p(-eps)

    # Biased-coin generator: heads rate ~ Beta(a, a), a fitted (a = 1 is the incumbent).
    log_a = pm.Normal("log_a", mu=0.0, sigma=1.0)
    a = pm.Deterministic("a", pt.exp(log_a))

    def _betaln(x, y):
        return pt.gammaln(x) + pt.gammaln(y) - pt.gammaln(x + y)

    seq_lbias_a = _betaln(seq_heads + a, seq_len - seq_heads + a) - _betaln(a, a)

    # Log marginal likelihood of each distinct sequence under the regular generators.
    motif_terms = []
    for p in PERIODS:
        free = pt.minimum(float(p), seq_len)
        motif_terms.append(
            -free * math.log(2.0)
            + (seq_cmp[p] - seq_mis[p]) * log_1m_eps
            + seq_mis[p] * log_eps
        )
    log_motif = pm.math.logsumexp(pt.stack(motif_terms, axis=0), axis=0, keepdims=False) - math.log(len(PERIODS))
    log_regular = pm.math.logsumexp(
        pt.stack([seq_lmark, seq_lbias_a, log_motif], axis=0), axis=0, keepdims=False
    ) - math.log(3.0)
    log_regular_diff = log_regular[idx_a] - log_regular[idx_b]

    # Person-specific decision sensitivity (non-centred log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    score_diff = switch_diff * logit_q[participant_id] - log_regular_diff

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta[participant_id] * score_diff)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
