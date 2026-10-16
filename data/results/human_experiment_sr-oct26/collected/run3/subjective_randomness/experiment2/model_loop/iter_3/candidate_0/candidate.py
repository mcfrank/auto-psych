"""Position-weighted switch impression in the second-order motif-suspicion judgment.

People judge a sequence random to the extent their own model of a fair coin
(person-specific believed switch rate, second-order switch belief) explains it
better than the regular generators (Markov coin, biased coin, repeating motif
with slips, person-specific motif suspicion), the evidence weighed per flip by a
fitted power of length. The single new claim: the felt switch count that the
believed switch rate acts on is read left to right with position-dependent
weight — transition t of m gets weight proportional to exp(lam * x_t), x_t from
-0.5 (first) to +0.5 (last), renormalised to sum to m — so lam > 0 means
switches/repeats near the end (recency) dominate the impression, lam < 0 the
start (primacy). lam = 0 recovers the position-blind incumbent.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)
MAXT = 7


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    log_markov = math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1)
    log_biased = _log_beta(h + 1, n - h + 1)
    trans = [1 if x != y else 0 for x, y in zip(seq, seq[1:])]
    ss = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 1)
    sr = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 0)
    m = max(n - 1, 1)
    tr_pad = [0.0] * MAXT
    x_pad = [0.0] * MAXT
    mask = [0.0] * MAXT
    for t, s in enumerate(trans):
        tr_pad[t] = float(s)
        x_pad[t] = (t / (len(trans) - 1) - 0.5) if len(trans) > 1 else 0.0
        mask[t] = 1.0
    out = {"tr": tr_pad, "x": x_pad, "mask": mask, "k": float(k), "lm": log_markov, "lb": log_biased, "alt2": float(ss - sr)}
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
    idx_a, idx_b, switch_diff, alt2_diff, pid, y = [], [], [], [], [], []
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
        alt2_diff.append(table[ia_ib[0]]["alt2"] - table[ia_ib[1]]["alt2"])
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    out = {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "switch_diff": np.asarray(switch_diff, dtype="float64"),
        "alt2_diff": np.asarray(alt2_diff, dtype="float64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "seq_lmark": np.asarray([t["lm"] for t in table], dtype="float64"),
        "seq_lbias": np.asarray([t["lb"] for t in table], dtype="float64"),
        "seq_len": np.asarray([t["n"] for t in table], dtype="float64"),
        "seq_tr": np.asarray([t["tr"] for t in table], dtype="float64"),
        "seq_x": np.asarray([t["x"] for t in table], dtype="float64"),
        "seq_mask": np.asarray([t["mask"] for t in table], dtype="float64"),
    }
    for p in PERIODS:
        out[f"seq_cmp{p}"] = np.asarray([t[f"c{p}"] for t in table], dtype="float64")
        out[f"seq_mis{p}"] = np.asarray([t[f"m{p}"] for t in table], dtype="float64")
    return out


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    alt2_diff = pm.Data("alt2_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
    seq_lbias = pm.Data("seq_lbias", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_tr = pm.Data("seq_tr", np.zeros((1, MAXT), dtype="float64"))
    seq_x = pm.Data("seq_x", np.zeros((1, MAXT), dtype="float64"))
    seq_mask = pm.Data("seq_mask", np.ones((1, MAXT), dtype="float64"))
    seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}


    # Slip rate of the repeating-motif generator.
    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    eps = pm.Deterministic("eps", pm.math.sigmoid(logit_eps))
    log_eps = pt.log(eps)
    log_1m_eps = pt.log1p(-eps)

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

    # Person-specific prior weight of the motif generator among the regular
    # explanations (weights 1 : 1 : exp(a_i)); the normaliser is the same for
    # both sequences of a pair, so it cancels in the difference.
    mu_a = pm.Normal("mu_a", mu=0.0, sigma=1.0)
    sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
    z_a = pm.Normal("z_a", mu=0.0, sigma=1.0, shape=400)
    a_motif = (mu_a + sigma_a * z_a)[participant_id]

    def _log_regular(idx):
        return pm.math.logsumexp(
            pt.stack([seq_lmark[idx], seq_lbias[idx], log_motif[idx] + a_motif], axis=0),
            axis=0,
            keepdims=False,
        )

    log_regular_diff = _log_regular(idx_a) - _log_regular(idx_b)

    # Person-specific decision sensitivity (non-centred log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    # Length normalisation of the evidence (gamma = 0: total evidence, 1: per flip).
    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    trial_len = seq_len[idx_a]
    length_scale = pt.exp(-gamma * pt.log(trial_len / 5.0))

    # Person-specific believed switch probability of a fair coin enters the
    # score as switch_diff * logit_q_i, scaled by that person's sensitivity
    # beta_i. Parameterised directly as the product w_i = beta_i * logit_q_i
    # (non-centred population; w_i < 0: the person believes chance is streaky)
    # to avoid the beta-by-logit_q funnel.
    mu_w = pm.Normal("mu_w", mu=0.0, sigma=1.0)
    sigma_w = pm.HalfNormal("sigma_w", sigma=0.5)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=400)
    w = mu_w + sigma_w * z_w

    # Second-order chance coin: shared shift d in the believed log-odds of
    # switching right after a switch (d < 0: chance rarely switches twice running).
    d_switch2 = pm.Normal("d_switch2", mu=0.0, sigma=1.0)

    # Position-weighted felt switch count (lam > 0: recency, < 0: primacy).
    lam = pm.Normal("lam", mu=0.0, sigma=1.5)
    raw_w = pt.exp(lam * seq_x) * seq_mask
    n_trans = pt.sum(seq_mask, axis=1)
    pos_w = raw_w * (n_trans / pt.sum(raw_w, axis=1))[:, None]
    k_felt = pt.sum(pos_w * seq_tr, axis=1)
    felt_switch_diff = k_felt[idx_a] - k_felt[idx_b]

    pid = participant_id
    log_chance2_diff = 0.5 * d_switch2 * alt2_diff
    eta = (w[pid] * felt_switch_diff + beta[pid] * (log_chance2_diff - log_regular_diff)) * length_scale

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
