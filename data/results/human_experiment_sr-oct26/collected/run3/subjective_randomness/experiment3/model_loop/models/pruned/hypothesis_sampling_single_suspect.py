"""Single-suspect hypothesis sampling in the chance-versus-regular randomness judgment.

People judge randomness as Bayesian inference: does their own second-order
picture of a fair coin explain a sequence better than the regular generators
(a switch-biased Markov coin, a heads-favoured trick coin, a short motif
repeated with slips)? The one distortion is a resource limit: instead of
averaging over the regular explanations (the normative logsumexp of their
marginal likelihoods), on each trial a person brings a single regular
explanation to mind, sampled with the generators' prior weights
(1 : 1 : exp(a_i), a_i the person's readiness to think of a motif), and judges
both sequences of the pair against that one. The choice probability is thus a
mixture, over which explanation came to mind, of the choice probabilities
computed against each explanation alone. A sequence regular under only one
explanation is condemned only on the trials where that explanation comes to
mind, which is most visible on very short pairs (where the explanations
disagree most). Everything else follows the current best account: person-
specific believed switch rate (as a sensitivity-weighted switch term), a
shared second-order switch belief, a heavy-tailed signed person sensitivity,
evidence weighed per flip by a fitted power of length and a person side habit.
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
    out = {"k": float(k), "lm": log_markov, "h": float(h), "alt2": float(ss - sr)}
    for p in PERIODS:
        out[f"c{p}"] = float(max(n - p, 0))
        out[f"m{p}"] = float(sum(1 for i in range(p, n) if seq[i] != seq[i - p]))
    out["n"] = float(n)
    return out


def prepare_observed(rows):
    """Table of distinct sequences plus per-trial indices."""
    index = {}
    table = []
    idx_a, idx_b, switch_diff, alt2_diff, pid, y = [], [], [], [], [], []
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
        switch_diff.append(table[ids[0]]["k"] - table[ids[1]]["k"])
        alt2_diff.append(table[ids[0]]["alt2"] - table[ids[1]]["alt2"])
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
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
    seq_heads = pm.Data("seq_heads", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}

    # Motif generator: slip rate, periods 1-4 equally likely.
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

    # Trick coin with a lopsided Beta(2 s, 2 (1 - s)) prior over its heads rate.
    logit_s = pm.Normal("logit_s", mu=0.0, sigma=1.0)
    s_heads = pm.Deterministic("s_heads", pm.math.sigmoid(logit_s))
    a_h = 2.0 * s_heads
    a_t = 2.0 - a_h
    seq_lbias = (
        pt.gammaln(seq_heads + a_h) + pt.gammaln(seq_len - seq_heads + a_t)
        - pt.gammaln(seq_len + 2.0)
        - pt.gammaln(a_h) - pt.gammaln(a_t) + math.lgamma(2.0)
    )

    pid = participant_id

    # Person-specific readiness to think of a motif: the probability that each
    # regular explanation comes to mind on a trial is softmax(0, 0, a_i).
    mu_a = pm.Normal("mu_a", mu=0.0, sigma=1.0)
    sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
    z_a = pm.Normal("z_a", mu=0.0, sigma=1.0, shape=400)
    a_motif = (mu_a + sigma_a * z_a)[pid]
    logits_mind = pt.stack([pt.zeros_like(a_motif), pt.zeros_like(a_motif), a_motif], axis=0)
    pi_mind = pt.exp(logits_mind - pm.math.logsumexp(logits_mind, axis=0, keepdims=True))

    # Heavy-tailed signed person sensitivity.
    mu_b = pm.Normal("mu_b", mu=1.0, sigma=0.5)
    sigma_b = pm.HalfNormal("sigma_b", sigma=0.5)
    z_beta = pm.StudentT("z_beta", nu=4.0, mu=0.0, sigma=1.0, shape=400)
    beta = mu_b + sigma_b * z_beta

    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    length_scale = pt.exp(-gamma * pt.log(seq_len[idx_a] / 5.0))

    # Person-specific believed switch rate (sensitivity-weighted, non-centred).
    mu_w = pm.Normal("mu_w", mu=0.0, sigma=1.0)
    sigma_w = pm.HalfNormal("sigma_w", sigma=0.5)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=400)
    w = mu_w + sigma_w * z_w

    d_switch2 = pm.Normal("d_switch2", mu=0.0, sigma=1.0)
    log_chance2_diff = 0.5 * d_switch2 * alt2_diff

    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.5)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=400)
    side = mu_side + sigma_side * z_side

    # Evidence difference against each single regular explanation.
    reg_diffs = [
        seq_lmark[idx_a] - seq_lmark[idx_b],
        seq_lbias[idx_a] - seq_lbias[idx_b],
        log_motif[idx_a] - log_motif[idx_b],
    ]
    base = w[pid] * switch_diff
    p_given = [
        pm.math.sigmoid((base + beta[pid] * (log_chance2_diff - rd)) * length_scale + side[pid])
        for rd in reg_diffs
    ]
    p_mix = pi_mind[0] * p_given[0] + pi_mind[1] * p_given[1] + pi_mind[2] * p_given[2]
    p_left = pm.Deterministic("p_left", pt.clip(p_mix, 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
