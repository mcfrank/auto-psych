"""Posterior-verdict contrast: the comparison is between saturating verdicts, not raw evidence.

People judge each sequence by how probable it is that a fair coin (believed to
switch with probability q, fitted) rather than a regular generator (a Markov
coin with unknown switch rate, a biased coin with unknown heads rate, or a
repeating short motif copied with slip rate eps) produced it, given a prior
log-odds alpha that a sequence is chance-made. The choice is driven by the
difference of these two posterior verdicts, P(chance | a) - P(chance | b), so
the same evidence difference is decisive between two ambiguous sequences and
nearly irrelevant between two clearly random (or two clearly regular) ones: a
sequence's effect on the choice depends on its partner. People differ in
decision sensitivity.
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
    out = {
        "n": float(n),
        "k": float(k),
        "lm": math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1),
        "lb": _log_beta(h + 1, n - h + 1),
    }
    for p in PERIODS:
        out[f"c{p}"] = float(max(n - p, 0))
        out[f"m{p}"] = float(sum(1 for i in range(p, n) if seq[i] != seq[i - p]))
    return out


def prepare_observed(rows):
    """Table of distinct sequences (generator statistics) plus per-trial indices."""
    index = {}
    table = []
    idx_a, idx_b, pid, y = [], [], [], []
    for r in rows:
        ids = []
        for key in ("sequence_a", "sequence_b"):
            seq = str(r[key]).strip().upper()
            if seq not in index:
                index[seq] = len(table)
                table.append(_summary(seq))
            ids.append(index[seq])
        idx_a.append(ids[0])
        idx_b.append(ids[1])
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    out = {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "seq_len": np.asarray([t["n"] for t in table], dtype="float64"),
        "seq_sw": np.asarray([t["k"] for t in table], dtype="float64"),
        "seq_lmark": np.asarray([t["lm"] for t in table], dtype="float64"),
        "seq_lbias": np.asarray([t["lb"] for t in table], dtype="float64"),
    }
    for p in PERIODS:
        out[f"seq_cmp{p}"] = np.asarray([t[f"c{p}"] for t in table], dtype="float64")
        out[f"seq_mis{p}"] = np.asarray([t[f"m{p}"] for t in table], dtype="float64")
    return out


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_sw = pm.Data("seq_sw", np.zeros(1, dtype="float64"))
    seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
    seq_lbias = pm.Data("seq_lbias", np.zeros(1, dtype="float64"))
    seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}

    # Believed switch probability of a fair coin.
    logit_q = pm.Normal("logit_q", mu=0.0, sigma=1.0)
    log_q = -pt.softplus(-logit_q)
    log_1mq = -pt.softplus(logit_q)

    # Slip rate of the repeating-motif generator.
    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    log_eps = -pt.softplus(-logit_eps)
    log_1m_eps = -pt.softplus(logit_eps)

    # Log likelihood of each distinct sequence under chance and under regularity.
    log_chance = math.log(0.5) + seq_sw * log_q + (seq_len - 1.0 - seq_sw) * log_1mq
    motif_terms = [
        -pt.minimum(float(p), seq_len) * math.log(2.0)
        + (seq_cmp[p] - seq_mis[p]) * log_1m_eps
        + seq_mis[p] * log_eps
        for p in PERIODS
    ]
    log_motif = pm.math.logsumexp(pt.stack(motif_terms, axis=0), axis=0, keepdims=False) - math.log(len(PERIODS))
    log_regular = pm.math.logsumexp(
        pt.stack([seq_lmark, seq_lbias, log_motif], axis=0), axis=0, keepdims=False
    ) - math.log(3.0)

    # Prior log-odds that a sequence is chance-made; the saturating verdict.
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
    verdict = pm.math.sigmoid(alpha + log_chance - log_regular)
    verdict_diff = verdict[idx_a] - verdict[idx_b]

    # Person-specific decision sensitivity (non-centred log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=1.5, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    # A fixed 0.1% response floor (numerical guard, not a fitted parameter):
    # keeps p_left off 0/1 for very decisive people so PSIS-LOO stays defined.
    p_left = pm.Deterministic(
        "p_left", 1e-3 + (1.0 - 2e-3) * pm.math.sigmoid(beta[participant_id] * verdict_diff)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
