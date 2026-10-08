"""Blurred encoding: randomness evidence averaged over possible mis-registrations of the flips.

People do not register a sequence of coin flips perfectly: they read each flip
with a fixed (fitted) chance e of mis-registering it, so their felt evidence that
a sequence came from chance (their own second-order, gambler's-fallacy picture of
a fair coin) rather than from a regular generator (a switch-biased coin, a biased
coin, a repeating short motif with slips) is that evidence averaged over the
versions of the sequence they might have registered, each weighted
e^d (1 - e)^(n - d) by its Hamming distance d. This strips exactly regular
sequences of much of their extreme evidence relative to one-slip neighbours and
dulls the switch preference most in very short pairs.

Everything else follows the current best account: person-specific believed
switch rate (as w_i = beta_i * logit q_i), second-order and run-length
gambler's terms, regular generators with a shared motif weight, evidence per
flip by a fitted power of length, person-specific sensitivity, side habit and
lapse share. The averaging runs over all 2^n sequences of each length (n = 2..8),
precomputed as constants; the blur is applied one flip position at a time.
"""

import itertools
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)
LENGTHS = tuple(range(2, 9))


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    trans = [1 if x != y else 0 for x, y in zip(seq, seq[1:])]
    ss = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 1)
    sr = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 0)
    gam = 0.0
    run = 1
    for x, y in zip(seq, seq[1:]):
        if run >= 3:
            gam += (run - 2) * (1.0 if x != y else -1.0)
        run = 1 if x != y else run + 1
    out = {
        "k": float(k),
        "lm": math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1),
        "lb": _log_beta(h + 1, n - h + 1),
        "alt2": float(ss - sr),
        "gam": gam,
        "n": float(n),
    }
    for p in PERIODS:
        out[f"c{p}"] = float(max(n - p, 0))
        out[f"m{p}"] = float(sum(1 for i in range(p, n) if seq[i] != seq[i - p]))
    return out


# Enumerate every sequence of each length; blocks are contiguous per length.
_SEQS = {n: ["".join(t) for t in itertools.product("HT", repeat=n)] for n in LENGTHS}
_INDEX = {}
_TABLE = []
_OFFSET = {}
for _n in LENGTHS:
    _OFFSET[_n] = len(_TABLE)
    for _s in _SEQS[_n]:
        _INDEX[_s] = len(_TABLE)
        _TABLE.append(_summary(_s))


def _col(key):
    return np.asarray([t[key] for t in _TABLE], dtype="float64")


def _seq_index(seq):
    seq = str(seq).strip().upper()
    if seq not in _INDEX:
        raise ValueError(f"sequence {seq!r} is not an H/T string of length 2-8")
    return _INDEX[seq]


def prepare_observed(rows):
    """Per-trial indices into the table of all H/T sequences of length 2-8."""
    return {
        "idx_a": np.asarray([_seq_index(r["sequence_a"]) for r in rows], dtype="int64"),
        "idx_b": np.asarray([_seq_index(r["sequence_b"]) for r in rows], dtype="int64"),
        "participant_id": np.asarray([int(r.get("participant_id", 0)) for r in rows], dtype="int64"),
        "chose_left": np.asarray([int(r.get("chose_left", 0)) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    pid = participant_id

    seq_len = pt.as_tensor_variable(_col("n"))

    # Regular generators: Markov switch coin, biased coin, repeating motif with slips.
    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    eps = pm.math.sigmoid(logit_eps)
    log_eps = pt.log(eps)
    log_1m_eps = pt.log1p(-eps)
    motif_terms = []
    for p in PERIODS:
        cmp_p = _col(f"c{p}")
        mis_p = _col(f"m{p}")
        free = np.minimum(float(p), _col("n"))
        motif_terms.append(-free * math.log(2.0) + (cmp_p - mis_p) * log_1m_eps + mis_p * log_eps)
    log_motif = pm.math.logsumexp(pt.stack(motif_terms, axis=0), axis=0, keepdims=False) - math.log(len(PERIODS))
    a_motif = pm.Normal("a_motif", mu=0.0, sigma=1.0)
    log_regular = pm.math.logsumexp(
        pt.stack([pt.as_tensor_variable(_col("lm")), pt.as_tensor_variable(_col("lb")), log_motif + a_motif], axis=0),
        axis=0,
        keepdims=False,
    )

    # Second-order and run-length gambler's beliefs in the chance coin.
    d_switch2 = pm.Normal("d_switch2", mu=0.0, sigma=1.0)
    g_run = pm.Normal("g_run", mu=0.0, sigma=1.0)
    evidence = 0.5 * d_switch2 * _col("alt2") + 0.5 * g_run * _col("gam") - log_regular

    # Blurred encoding: each flip mis-registered with probability e_read; the
    # felt switch count and evidence are averaged over registered versions.
    logit_read = pm.Normal("logit_read", mu=-3.0, sigma=1.0)
    e_read = pm.Deterministic("e_read", pm.math.sigmoid(logit_read))
    k_all = pt.as_tensor_variable(_col("k"))

    def _blur(v, n):
        # Independent per-flip misreading is a Kronecker product of 2x2 kernels:
        # apply it one flip position (array axis) at a time.
        v = v.reshape((2,) * n)
        for ax in range(n):
            flipped = pt.flip(v, axis=ax)
            v = (1.0 - e_read) * v + e_read * flipped
        return v.reshape((2 ** n,))

    blur_k, blur_ev = [], []
    for n in LENGTHS:
        lo, hi = _OFFSET[n], _OFFSET[n] + 2 ** n
        blur_k.append(_blur(k_all[lo:hi], n))
        blur_ev.append(_blur(evidence[lo:hi], n))
    blur_k = pt.concatenate(blur_k)
    blur_ev = pt.concatenate(blur_ev)

    switch_diff = blur_k[idx_a] - blur_k[idx_b]
    ev_diff = blur_ev[idx_a] - blur_ev[idx_b]

    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    length_scale = pt.exp(-gamma * pt.log(seq_len[idx_a] / 5.0))

    mu_w = pm.Normal("mu_w", mu=0.0, sigma=1.0)
    sigma_w = pm.HalfNormal("sigma_w", sigma=0.5)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=400)
    w = mu_w + sigma_w * z_w

    eta = (w[pid] * switch_diff + beta[pid] * ev_diff) * length_scale

    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.5)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=400)
    side = mu_side + sigma_side * z_side

    mu_lapse = pm.Normal("mu_lapse", mu=-3.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=400)
    lapse = pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse)[pid]

    p_judge = pm.math.sigmoid(eta + side[pid])
    p_left = pm.Deterministic(
        "p_left", pt.clip((1.0 - lapse) * p_judge + 0.5 * lapse, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
