"""Attention gradient over the reading order in sequential chance-vs-regular evidence.

People read a sequence flip by flip and accumulate evidence for "fair coin"
against the regular generators they suspect (a switch-biased Markov coin, a
biased coin, a repeating short motif with slips; unknowns integrated out by
sequential Laplace prediction, person-specific believed switch rate of chance
and person-specific motif suspicion), but each flip's contribution is weighted
by an attention gradient along the reading order, w_t proportional to
exp(lam * (u_t - 0.5)) with u_t the relative position, normalised to mean 1
(lam > 0 recency, lam < 0 primacy). Where in the sequence a streak, imbalance
or pattern break occurs therefore changes how random it looks.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)
L = 8


def _seq_table(seq):
    seq = seq.strip().upper()
    n = len(seq)
    mask = np.zeros(L)
    u = np.zeros(L)
    sw = np.zeros(L)
    lmark = np.zeros(L)
    lbias = np.zeros(L)
    free = np.zeros((len(PERIODS), L))
    match = np.zeros((len(PERIODS), L))
    mis = np.zeros((len(PERIODS), L))
    k = 0
    h = 0
    for t in range(n):
        mask[t] = 1.0
        u[t] = t / (n - 1) if n > 1 else 0.5
        heads = seq[t] == "H"
        # biased coin, sequential Laplace prediction of heads
        p_h = (h + 1.0) / (t + 2.0)
        lbias[t] = math.log(p_h if heads else 1.0 - p_h)
        if t == 0:
            lmark[t] = math.log(0.5)
        else:
            s = seq[t] != seq[t - 1]
            sw[t] = 1.0 if s else 0.0
            p_s = (k + 1.0) / ((t - 1) + 2.0)
            lmark[t] = math.log(p_s if s else 1.0 - p_s)
            k += int(s)
        h += int(heads)
        for j, p in enumerate(PERIODS):
            if t < p:
                free[j, t] = 1.0
            elif seq[t] == seq[t - p]:
                match[j, t] = 1.0
            else:
                mis[j, t] = 1.0
    return dict(n=float(n), mask=mask, u=u, sw=sw, lmark=lmark, lbias=lbias,
                free=free, match=match, mis=mis)


def prepare_observed(rows):
    index, table = {}, []
    idx_a, idx_b, pid, y = [], [], [], []
    for r in rows:
        ids = []
        for key in ("sequence_a", "sequence_b"):
            seq = str(r[key]).strip().upper()
            if seq not in index:
                index[seq] = len(table)
                table.append(_seq_table(seq))
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
    }
    for name in ("mask", "u", "sw", "lmark", "lbias"):
        out[f"seq_{name}"] = np.asarray([t[name] for t in table], dtype="float64")
    for name in ("free", "match", "mis"):
        out[f"seq_{name}"] = np.asarray([t[name] for t in table], dtype="float64")
    return out


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_mask = pm.Data("seq_mask", np.ones((1, L)))
    seq_u = pm.Data("seq_u", np.zeros((1, L)))
    seq_sw = pm.Data("seq_sw", np.zeros((1, L)))
    seq_lmark = pm.Data("seq_lmark", np.zeros((1, L)))
    seq_lbias = pm.Data("seq_lbias", np.zeros((1, L)))
    seq_free = pm.Data("seq_free", np.zeros((1, len(PERIODS), L)))
    seq_match = pm.Data("seq_match", np.zeros((1, len(PERIODS), L)))
    seq_mis = pm.Data("seq_mis", np.zeros((1, len(PERIODS), L)))

    # Attention gradient along the reading order (lam > 0 recency, < 0 primacy).
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)
    raw_w = pt.exp(lam * (seq_u - 0.5)) * seq_mask
    att = raw_w * seq_mask.sum(axis=1, keepdims=True) / raw_w.sum(axis=1, keepdims=True)

    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    eps = pm.math.sigmoid(logit_eps)
    log_eps = pt.log(eps)
    log_1m_eps = pt.log1p(-eps)

    # Attention-weighted log likelihood of each distinct sequence per generator.
    w_sw = (att * seq_sw).sum(axis=1)
    l_mark = (att * seq_lmark).sum(axis=1)
    l_bias = (att * seq_lbias).sum(axis=1)
    att3 = att[:, None, :]
    l_motif_p = (
        (att3 * seq_free).sum(axis=2) * (-math.log(2.0))
        + (att3 * seq_match).sum(axis=2) * log_1m_eps
        + (att3 * seq_mis).sum(axis=2) * log_eps
    )
    l_motif = pm.math.logsumexp(l_motif_p, axis=1, keepdims=False) - math.log(len(PERIODS))

    # Person-specific motif suspicion (weights 1 : 1 : exp(a_i)).
    mu_a = pm.Normal("mu_a", mu=0.0, sigma=1.0)
    sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
    z_a = pm.Normal("z_a", mu=0.0, sigma=1.0, shape=400)
    a_motif = (mu_a + sigma_a * z_a)[participant_id]

    def _log_regular(idx):
        return pm.math.logsumexp(
            pt.stack([l_mark[idx], l_bias[idx], l_motif[idx] + a_motif], axis=0),
            axis=0,
            keepdims=False,
        )

    log_regular_diff = _log_regular(idx_a) - _log_regular(idx_b)

    # Person-specific decision sensitivity.
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    length_scale = pt.exp(-gamma * pt.log(seq_len[idx_a] / 5.0))

    # Person-specific believed switch rate of chance, as beta_i * logit q_i.
    mu_w = pm.Normal("mu_w", mu=0.0, sigma=1.0)
    sigma_w = pm.HalfNormal("sigma_w", sigma=0.5)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=400)
    w = mu_w + sigma_w * z_w

    pid = participant_id
    switch_diff = w_sw[idx_a] - w_sw[idx_b]
    eta = (w[pid] * switch_diff - beta[pid] * log_regular_diff) * length_scale

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
