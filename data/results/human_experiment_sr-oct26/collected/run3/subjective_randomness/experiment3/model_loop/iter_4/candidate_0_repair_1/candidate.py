"""Tally-correcting chance coin weighed against the regular generators.

People's picture of a fair coin keeps a running tally of heads minus tails and
expects chance to correct the lead: whenever one face is ahead by |D| flips,
the believed log-odds that the next flip favours the trailing face rise by a
shared fitted r per unit of lead. Each flip read while the tally is unbalanced
contributes (r / 2) * |D| * (+1 if it lands on the trailing face, -1 if it
extends the lead) to the chance log-likelihood, so a sequence whose tally
drifts and is promptly corrected looks random, a growing lead looks non-random,
and breaking a long streak (correcting a large lead) is strongly rewarded: the
path to the final count matters, not the count. Everything else is the current
best account (`streak_weary_chance_pseudoflip`): person-specific believed switch
rate and a second-order switch belief in the chance coin, weighed against a
switch-biased coin, a heads-favoured trick coin and a repeating motif with slips
(person-specific motif suspicion), per-flip evidence with pseudo-flip dilution,
heavy-tailed signed person sensitivity (around a positive group mean) and a
side habit. The tally correction replaces that model's run-length gambler's
belief.
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
    trans = [1 if x != y else 0 for x, y in zip(seq, seq[1:])]
    ss = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 1)
    sr = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 0)
    # Running tally: each flip read while one face leads by |D| scores
    # +|D| if it lands on the trailing face, -|D| if it extends the lead.
    gam = 0.0
    lead = 0
    for x in seq:
        step = 1 if x == "H" else -1
        if lead != 0:
            gam += abs(lead) * (1.0 if step * lead < 0 else -1.0)
        lead += step
    out = {"k": float(k), "lm": log_markov, "h": float(h), "alt2": float(ss - sr), "gam": gam}
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
    idx_a, idx_b, switch_diff, alt2_diff, gam_diff, pid, y = [], [], [], [], [], [], []
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
        gam_diff.append(table[ia_ib[0]]["gam"] - table[ia_ib[1]]["gam"])
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    out = {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "switch_diff": np.asarray(switch_diff, dtype="float64"),
        "alt2_diff": np.asarray(alt2_diff, dtype="float64"),
        "gam_diff": np.asarray(gam_diff, dtype="float64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "seq_lmark": np.asarray([t["lm"] for t in table], dtype="float64"),
        "seq_heads": np.asarray([t["h"] for t in table], dtype="float64"),
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
    alt2_diff = pm.Data("alt2_diff", np.zeros(1, dtype="float64"))
    gam_diff = pm.Data("gam_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
    seq_heads = pm.Data("seq_heads", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}


    # Slip rate of the repeating-motif generator.
    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    eps = pm.Deterministic("eps", pm.math.sigmoid(logit_eps))
    log_eps = pt.log(eps)
    log_1m_eps = pt.log1p(-eps)

    # Trick coin with a lopsided prior over its heads rate: Beta(2 s, 2 (1 - s)),
    # s > 0.5 means coins rigged towards heads are suspected more.
    # Prior tightened (sigma 0.5): with sigma 1 a degenerate mode at s ~ 0.02
    # (a near-certain tails-rigged coin) trapped a chain; the data put s near 0.5.
    logit_s = pm.Normal("logit_s", mu=0.0, sigma=0.5)
    s_heads = pm.Deterministic("s_heads", pm.math.sigmoid(logit_s))
    a_h = 2.0 * s_heads
    a_t = 2.0 - a_h
    seq_lbias = (
        pt.gammaln(seq_heads + a_h) + pt.gammaln(seq_len - seq_heads + a_t)
        - pt.gammaln(seq_len + 2.0)
        - pt.gammaln(a_h) - pt.gammaln(a_t) + math.lgamma(2.0)
    )

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

    # Person-specific signed sensitivity (non-centred, heavy-tailed population):
    # most people near the group mean, a few near zero or reversed.
    # The group's typical sensitivity is positive (log-normal): a reversed
    # group mean let the tally term and the regular evidence flip sign together,
    # a second mode that trapped a chain. Individuals may still reverse.
    log_mu_b = pm.Normal("log_mu_b", mu=0.0, sigma=0.5)
    mu_b = pm.Deterministic("mu_b", pt.exp(log_mu_b))
    sigma_b = pm.HalfNormal("sigma_b", sigma=0.5)
    z_beta = pm.StudentT("z_beta", nu=4.0, mu=0.0, sigma=1.0, shape=400)
    beta = mu_b + sigma_b * z_beta

    # Length normalisation of the evidence (gamma = 0: total evidence, 1: per flip).
    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    trial_len = seq_len[idx_a]
    # Pseudo-flip dilution: c imagined unremarkable flips pad the length.
    log_c = pm.Normal("log_c", mu=0.5, sigma=1.0)
    c_pad = pm.Deterministic("c_pad", pt.exp(log_c))
    length_scale = pt.exp(-gamma * pt.log((trial_len + c_pad) / (5.0 + c_pad)))

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

    pid = participant_id
    # Tally-correcting chance coin: shared rise r in the believed log-odds that
    # the next flip favours the trailing face, per unit of current lead.
    g_run = pm.Normal("r_tally", mu=0.0, sigma=0.3)
    log_chance2_diff = 0.5 * d_switch2 * alt2_diff + 0.5 * g_run * gam_diff
    eta = (w[pid] * switch_diff + beta[pid] * (log_chance2_diff - log_regular_diff)) * length_scale

    # Person-specific side habit (non-centred population, shared mean bias).
    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.5)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=400)
    side = mu_side + sigma_side * z_side

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta + side[pid]))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
