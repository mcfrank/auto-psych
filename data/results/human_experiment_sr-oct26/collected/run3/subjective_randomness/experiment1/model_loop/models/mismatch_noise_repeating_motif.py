"""Paired comparison with mismatch-scaled noise, against regular generators that include a repeating motif.

Refinement of `mismatch_noise_paired_comparison`. People compare the two
sequences flip against flip: matching positions cancel, and the comparison
noise grows with the number of mismatching positions. Each sequence's
randomness evidence is the Bayesian "fair coin (believed to switch with
probability q, fitted) vs regular generator" score. The single change: the
regular generators now include a repeating-motif process (period 1-4, motif
uniform, each flip copies the one p back with slip rate eps, fitted), beside
the Markov coin with unknown switch rate and the coin with unknown bias.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    out = {
        "k": float(k),
        "lm": math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1),
        "lb": _log_beta(h + 1, n - h + 1),
        "n": float(n),
    }
    for p in PERIODS:
        out[f"c{p}"] = float(max(n - p, 0))
        out[f"m{p}"] = float(sum(1 for i in range(p, n) if seq[i] != seq[i - p]))
    return out


def prepare_observed(rows):
    """Distinct-sequence table plus per-trial indices, switch difference and mismatch count."""
    index = {}
    table = []
    idx_a, idx_b, switch_diff, n_mis, pid, y = [], [], [], [], [], []
    for r in rows:
        a = str(r["sequence_a"]).strip().upper()
        b = str(r["sequence_b"]).strip().upper()
        ids = []
        for seq in (a, b):
            if seq not in index:
                index[seq] = len(table)
                table.append(_summary(seq))
            ids.append(index[seq])
        idx_a.append(ids[0])
        idx_b.append(ids[1])
        switch_diff.append(table[ids[0]]["k"] - table[ids[1]]["k"])
        n_mis.append(float(sum(1 for x, z in zip(a, b) if x != z)))
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    out = {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "switch_diff": np.asarray(switch_diff, dtype="float64"),
        "n_mismatch": np.asarray(n_mis, dtype="float64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "seq_lmark": np.asarray([t["lm"] for t in table], dtype="float64"),
        "seq_lbias": np.asarray([t["lb"] for t in table], dtype="float64"),
        "seq_len": np.asarray([t["n"] for t in table], dtype="float64"),
    }
    for p in PERIODS:
        out[f"seq_cmp{p}"] = np.asarray([t[f"c{p}"] for t in table], dtype="float64")
        out[f"seq_mis{p}"] = np.asarray([t[f"m{p}"] for t in table], dtype="float64")
    return out


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    n_mismatch = pm.Data("n_mismatch", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
    seq_lbias = pm.Data("seq_lbias", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}

    # Believed switch probability of a fair coin (logit scale).
    logit_q = pm.Normal("logit_q", mu=0.0, sigma=1.0)

    # Slip rate of the repeating-motif generator.
    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    eps = pm.math.sigmoid(logit_eps)
    log_eps = pt.log(eps)
    log_1m_eps = pt.log1p(-eps)

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
        pt.stack([seq_lmark, seq_lbias, log_motif], axis=0), axis=0, keepdims=False
    ) - math.log(3.0)
    log_regular_diff = log_regular[idx_a] - log_regular[idx_b]

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
