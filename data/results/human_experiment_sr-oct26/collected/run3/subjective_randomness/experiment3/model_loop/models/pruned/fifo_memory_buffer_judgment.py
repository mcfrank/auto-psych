"""Limited first-in-first-out working-memory buffer for the randomness judgment.

People read a sequence flip by flip into a working-memory buffer of limited
capacity K; once full, each new flip displaces the oldest, so the judgment is
made only on the last min(K, n) flips. K varies from trial to trial around a
fitted typical span kappa in (2, 9) (a discretised normal with sd 1.5 over K = 2..8, the
top category holding every capacity >= 8). On the held flips the judgment is the
current best account's: how much better the person's own second-order picture
of a fair coin explains them than the regular generators (switch-biased coin,
heads-leaning trick coin, repeating motif with slips), weighed per held flip, with person-specific signed sensitivity and
side habit. The felt evidence is the average over buffer capacities (computed
once per distinct pair). The motif generator's prior weight is shared.
Streaks, imbalances and patterns confined to the start of a long sequence are
forgotten; the same features at its end count fully.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)
CAPS = (2, 3, 4, 5, 6, 7, 8)


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    trans = [1 if x != y else 0 for x, y in zip(seq, seq[1:])]
    ss = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 1)
    sr = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 0)
    out = {
        "k": float(k),
        "lm": math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1),
        "h": float(seq.count("H")),
        "alt2": float(ss - sr),
        "n": float(n),
    }
    for p in PERIODS:
        out[f"c{p}"] = float(max(n - p, 0))
        out[f"m{p}"] = float(sum(1 for i in range(p, n) if seq[i] != seq[i - p]))
    return out


def prepare_observed(rows):
    """Distinct held windows, and for each distinct pair its windows at every capacity."""
    index, table = {}, []
    pair_index, pairs = {}, []

    def lookup(seq):
        if seq not in index:
            index[seq] = len(table)
            table.append(_summary(seq))
        return index[seq]

    pair_idx = []
    for r in rows:
        key = (str(r["sequence_a"]).strip().upper(), str(r["sequence_b"]).strip().upper())
        if key not in pair_index:
            pair_index[key] = len(pairs)
            pairs.append(key)
        pair_idx.append(pair_index[key])

    ia, ib, sd, ad, ln = [], [], [], [], []
    for cap in CAPS:
        for a, b in pairs:
            i, j = lookup(a[-cap:]), lookup(b[-cap:])
            ia.append(i)
            ib.append(j)
            sd.append(table[i]["k"] - table[j]["k"])
            ad.append(table[i]["alt2"] - table[j]["alt2"])
            ln.append(table[i]["n"])
    out = {
        "pair_idx": np.asarray(pair_idx, dtype="int64"),
        "win_a": np.asarray(ia, dtype="int64"),
        "win_b": np.asarray(ib, dtype="int64"),
        "switch_diff": np.asarray(sd, dtype="float64"),
        "alt2_diff": np.asarray(ad, dtype="float64"),
        "held_len": np.asarray(ln, dtype="float64"),
        "participant_id": np.asarray([int(r.get("participant_id", 0)) for r in rows], dtype="int64"),
        "chose_left": np.asarray([int(r.get("chose_left", 0)) for r in rows], dtype="int64"),
        "seq_lmark": np.asarray([t["lm"] for t in table], dtype="float64"),
        "seq_heads": np.asarray([t["h"] for t in table], dtype="float64"),
        "seq_len": np.asarray([t["n"] for t in table], dtype="float64"),
    }
    for p in PERIODS:
        out[f"seq_cmp{p}"] = np.asarray([t[f"c{p}"] for t in table], dtype="float64")
        out[f"seq_mis{p}"] = np.asarray([t[f"m{p}"] for t in table], dtype="float64")
    return out


NC = len(CAPS)

with pm.Model() as model:
    pair_idx = pm.Data("pair_idx", np.zeros(1, dtype="int64"))
    win_a = pm.Data("win_a", np.zeros(NC, dtype="int64"))
    win_b = pm.Data("win_b", np.zeros(NC, dtype="int64"))
    switch_diff = pm.Data("switch_diff", np.zeros(NC, dtype="float64"))
    alt2_diff = pm.Data("alt2_diff", np.zeros(NC, dtype="float64"))
    held_len = pm.Data("held_len", np.full(NC, 2.0))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
    seq_heads = pm.Data("seq_heads", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}

    # Per-capacity, per-distinct-pair arrays (capacity-major).
    wa = win_a.reshape((NC, -1))
    wb = win_b.reshape((NC, -1))
    sdiff = switch_diff.reshape((NC, -1))
    adiff = alt2_diff.reshape((NC, -1))
    hlen = held_len.reshape((NC, -1))

    # Buffer capacity: discretised normal (sd 1.5) around the typical span
    # kappa, kept inside (2, 9) by a logistic map so it cannot drift onto the
    # plateau where every capacity holds the whole sequence; the top category
    # holds every capacity >= 8 (the whole sequence held).
    z_kappa = pm.Normal("z_kappa", mu=0.0, sigma=1.0)
    kappa = pm.Deterministic("kappa", 2.0 + 7.0 * pm.math.sigmoid(z_kappa))
    edges = np.array([c + 0.5 for c in CAPS[:-1]], dtype="float64")
    cdf = 0.5 * (1.0 + pt.erf((edges - kappa) / (1.5 * math.sqrt(2.0))))
    cdf_full = pt.concatenate([pt.zeros(1), cdf, pt.ones(1)])
    w_cap = pt.clip(cdf_full[1:] - cdf_full[:-1], 1e-12, 1.0)
    w_cap = (w_cap / pt.sum(w_cap))[:, None]

    # Regular generators on each distinct held window.
    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    eps = pm.Deterministic("eps", pm.math.sigmoid(logit_eps))
    log_eps = pt.log(eps)
    log_1m_eps = pt.log1p(-eps)

    logit_s = pm.Normal("logit_s", mu=0.0, sigma=1.0)
    s_heads = pm.Deterministic("s_heads", pm.math.sigmoid(logit_s))
    a_h = 2.0 * s_heads
    a_t = 2.0 - a_h
    seq_lbias = (
        pt.gammaln(seq_heads + a_h) + pt.gammaln(seq_len - seq_heads + a_t)
        - pt.gammaln(seq_len + 2.0)
        - pt.gammaln(a_h) - pt.gammaln(a_t) + math.lgamma(2.0)
    )

    motif_terms = []
    for p in PERIODS:
        free = pt.minimum(float(p), seq_len)
        motif_terms.append(
            -free * math.log(2.0)
            + (seq_cmp[p] - seq_mis[p]) * log_1m_eps
            + seq_mis[p] * log_eps
        )
    log_motif = pm.math.logsumexp(pt.stack(motif_terms, axis=0), axis=0, keepdims=False) - math.log(len(PERIODS))

    # Shared prior weight of the motif generator among the regular explanations.
    a_motif = pm.Normal("a_motif", mu=0.0, sigma=1.0)
    log_regular = pm.math.logsumexp(
        pt.stack([seq_lmark, seq_lbias, log_motif + a_motif], axis=0), axis=0, keepdims=False
    )

    # Evidence weighed per held flip, averaged over buffer capacities, per distinct pair.
    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    ls = pt.exp(-gamma * pt.log(hlen / 5.0)) * w_cap
    d_switch2 = pm.Normal("d_switch2", mu=0.0, sigma=1.0)
    S_pair = pt.sum(ls * sdiff, axis=0)
    E_pair = pt.sum(ls * (0.5 * d_switch2 * adiff - (log_regular[wa] - log_regular[wb])), axis=0)

    pid = participant_id

    # Person-specific signed sensitivity (heavy-tailed population).
    mu_b = pm.Normal("mu_b", mu=1.0, sigma=0.5)
    sigma_b = pm.HalfNormal("sigma_b", sigma=0.5)
    z_beta = pm.StudentT("z_beta", nu=4.0, mu=0.0, sigma=1.0, shape=400)
    beta = mu_b + sigma_b * z_beta

    # Person-specific believed switch rate of chance, times sensitivity.
    mu_w = pm.Normal("mu_w", mu=0.0, sigma=1.0)
    sigma_w = pm.HalfNormal("sigma_w", sigma=0.5)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=400)
    w = mu_w + sigma_w * z_w

    # Person-specific side habit.
    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.5)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=400)
    side = mu_side + sigma_side * z_side

    eta = w[pid] * S_pair[pair_idx] + beta[pid] * E_pair[pair_idx] + side[pid]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
