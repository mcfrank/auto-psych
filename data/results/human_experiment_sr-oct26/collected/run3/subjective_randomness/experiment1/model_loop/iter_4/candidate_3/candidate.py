"""Bayesian chance-vs-regular randomness judgment with a gambler's-fallacy chance model.

Refinement of `length_normalised_chance_vs_motif`. A sequence looks random to
the extent people's model of a fair coin explains it better than the regular
generators (Markov coin with unknown switch rate, biased coin with unknown
heads rate, repeating motif of period 1-4 with slip rate eps), with the
evidence divided by a fitted power of sequence length and person-specific
sensitivity. Single change: the chance model is a gambler's-fallacy coin whose
switch probability is sigmoid(a + b * (r - 1)), where r is the length of the
current run, so long streaks are improbable under chance beyond their switch
count, and perfect alternation earns only the baseline switch rate.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)
MAX_RUN = 7


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    out = {
        "lm": math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1),
        "lb": _log_beta(h + 1, n - h + 1),
        "n": float(n),
    }
    for p in PERIODS:
        out[f"c{p}"] = float(max(n - p, 0))
        out[f"m{p}"] = float(sum(1 for i in range(p, n) if seq[i] != seq[i - p]))
    # Transitions by current run length: switches and stays at run length r.
    sw = [0.0] * MAX_RUN
    st = [0.0] * MAX_RUN
    run = 1
    for i in range(1, n):
        r = min(run, MAX_RUN) - 1
        if seq[i] != seq[i - 1]:
            sw[r] += 1.0
            run = 1
        else:
            st[r] += 1.0
            run += 1
    out["sw"] = sw
    out["st"] = st
    return out


def prepare_observed(rows):
    """Table of distinct sequences plus per-trial indices."""
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
        "seq_lmark": np.asarray([t["lm"] for t in table], dtype="float64"),
        "seq_lbias": np.asarray([t["lb"] for t in table], dtype="float64"),
        "seq_len": np.asarray([t["n"] for t in table], dtype="float64"),
        "seq_sw": np.asarray([t["sw"] for t in table], dtype="float64"),
        "seq_st": np.asarray([t["st"] for t in table], dtype="float64"),
    }
    for p in PERIODS:
        out[f"seq_cmp{p}"] = np.asarray([t[f"c{p}"] for t in table], dtype="float64")
        out[f"seq_mis{p}"] = np.asarray([t[f"m{p}"] for t in table], dtype="float64")
    return out


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
    seq_lbias = pm.Data("seq_lbias", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_sw = pm.Data("seq_sw", np.zeros((1, MAX_RUN), dtype="float64"))
    seq_st = pm.Data("seq_st", np.zeros((1, MAX_RUN), dtype="float64"))
    seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}

    # Gambler's-fallacy chance model: logit P(switch | run r) = a + b * (r - 1).
    logit_q = pm.Normal("logit_q", mu=0.0, sigma=1.0)
    gf_slope = pm.Normal("gf_slope", mu=0.0, sigma=0.5)
    eta = logit_q + gf_slope * pt.arange(MAX_RUN, dtype="float64")
    log_chance = (
        seq_sw * (-pt.softplus(-eta))[None, :]
        + seq_st * (-pt.softplus(eta))[None, :]
    ).sum(axis=1) + math.log(0.5)

    # Repeating-motif generator slip rate.
    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    eps = pm.Deterministic("eps", pm.math.sigmoid(logit_eps))
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

    evidence = log_chance - log_regular
    evidence_diff = evidence[idx_a] - evidence[idx_b]

    # Person-specific decision sensitivity (non-centred log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    # Length normalisation of the evidence.
    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    length_scale = pt.exp(-gamma * pt.log(seq_len[idx_a] / 5.0))

    p_left = pm.Deterministic(
        "p_left", pt.clip(
            pm.math.sigmoid(beta[participant_id] * evidence_diff * length_scale),
            1e-6,
            1 - 1e-6,
        )
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
