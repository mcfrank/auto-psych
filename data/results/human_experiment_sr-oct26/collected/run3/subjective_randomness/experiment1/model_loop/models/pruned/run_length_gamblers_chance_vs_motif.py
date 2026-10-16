"""Bayesian randomness judgment with a gambler's-fallacy model of chance.

Refinement of `length_normalised_chance_vs_motif`: a sequence looks random to
the extent people's model of a fair coin explains it better than the regular
generators (switch-biased Markov coin, biased coin, repeating short motif with
slips; unknowns integrated out), with the log evidence difference divided by a
fitted power of length. The single change: people's chance model is a
gambler's-fallacy coin whose switch probability grows with the length of the
current run (logit q_r = logit_q + delta * (r - 1), delta >= 0), so long
streaks are improbable under "chance" beyond what their switch count implies,
while alternation (runs of one) is what chance is expected to produce.
People differ in decision sensitivity.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)
RUN_LENGTHS = (1, 2, 3, 4)  # 4 means "4 or more"


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    log_markov = math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1)
    log_biased = _log_beta(h + 1, n - h + 1)
    out = {"lm": log_markov, "lb": log_biased, "n": float(n)}
    for p in PERIODS:
        out[f"c{p}"] = float(max(n - p, 0))
        out[f"m{p}"] = float(sum(1 for i in range(p, n) if seq[i] != seq[i - p]))
    sw = {r: 0.0 for r in RUN_LENGTHS}
    rp = {r: 0.0 for r in RUN_LENGTHS}
    run = 1
    for i in range(1, n):
        r = min(run, RUN_LENGTHS[-1])
        if seq[i] != seq[i - 1]:
            sw[r] += 1.0
            run = 1
        else:
            rp[r] += 1.0
            run += 1
    for r in RUN_LENGTHS:
        out[f"sw{r}"] = sw[r]
        out[f"rp{r}"] = rp[r]
    return out


def prepare_observed(rows):
    """Table of distinct sequences plus per-trial indices."""
    index = {}
    table = []
    idx_a, idx_b, pid, y = [], [], [], []
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
    }
    for p in PERIODS:
        out[f"seq_cmp{p}"] = np.asarray([t[f"c{p}"] for t in table], dtype="float64")
        out[f"seq_mis{p}"] = np.asarray([t[f"m{p}"] for t in table], dtype="float64")
    for r in RUN_LENGTHS:
        out[f"seq_sw{r}"] = np.asarray([t[f"sw{r}"] for t in table], dtype="float64")
        out[f"seq_rp{r}"] = np.asarray([t[f"rp{r}"] for t in table], dtype="float64")
    return out


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
    seq_lbias = pm.Data("seq_lbias", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_sw = {r: pm.Data(f"seq_sw{r}", np.zeros(1, dtype="float64")) for r in RUN_LENGTHS}
    seq_rp = {r: pm.Data(f"seq_rp{r}", np.zeros(1, dtype="float64")) for r in RUN_LENGTHS}

    # Gambler's-fallacy chance model: switch probability after a run of length r.
    logit_q = pm.Normal("logit_q", mu=0.0, sigma=1.0)
    delta = pm.HalfNormal("delta", sigma=1.0)
    log_chance = 0.0
    for r in RUN_LENGTHS:
        a_r = logit_q + delta * (r - 1)
        log_chance = log_chance + seq_sw[r] * pt.log(pm.math.sigmoid(a_r)) + seq_rp[r] * pt.log(
            pm.math.sigmoid(-a_r)
        )

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

    # Randomness evidence per sequence (first flip's 1/2 cancels within a pair).
    evidence = log_chance - log_regular
    evidence_diff = evidence[idx_a] - evidence[idx_b]

    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    trial_len = seq_len[idx_a]
    length_scale = pt.exp(-gamma * pt.log(trial_len / 5.0))

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta[participant_id] * evidence_diff * length_scale)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
