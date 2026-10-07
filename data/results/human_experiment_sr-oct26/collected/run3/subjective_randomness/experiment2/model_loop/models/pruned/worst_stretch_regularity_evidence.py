"""Most damning stretch: the maximum over sub-sequences of regular-vs-chance evidence.

People judge randomness by the single most suspicious stretch of a sequence:
they scan every contiguous stretch of three or more flips (the whole sequence
when it is shorter), ask how much better a regular generator (a switch-biased
Markov coin, a biased coin, or a short motif copied with slips; unknowns
integrated out, equal prior weights) explains that stretch than their own fair
coin (with a fitted believed switch rate), and the largest such log
Bayes factor anywhere (a soft maximum over stretches) in the sequence alone sets how non-random it looks. The
sequence whose worst stretch is less damning is chosen as more random, with
shared decisiveness and the evidence weighed by (n / 5) ** -gamma.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)
MIN_WINDOW = 3
MAX_WINDOWS = 21  # windows of length >= 3 in an 8-flip sequence


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _windows(seq):
    n = len(seq)
    if n < MIN_WINDOW:
        return [seq]
    return [seq[i:j] for i in range(n) for j in range(i + MIN_WINDOW, n + 1)]


def _window_stats(w):
    L = len(w)
    k = sum(1 for x, y in zip(w, w[1:]) if x != y)
    h = w.count("H")
    out = {
        "L": float(L),
        "k": float(k),
        "lm": math.log(0.5) + _log_beta(k + 1, (L - 1 - k) + 1),
        "lb": _log_beta(h + 1, L - h + 1),
    }
    for p in PERIODS:
        out[f"c{p}"] = float(max(L - p, 0))
        out[f"m{p}"] = float(sum(1 for i in range(p, L) if w[i] != w[i - p]))
        out[f"f{p}"] = float(min(p, L))
    return out


def prepare_observed(rows):
    """Table of distinct sequences x padded window slots, plus per-trial indices."""
    index, table = {}, []
    idx_a, idx_b, y = [], [], []
    for r in rows:
        ii = []
        for key in ("sequence_a", "sequence_b"):
            seq = str(r[key]).strip().upper()
            if seq not in index:
                index[seq] = len(table)
                table.append((len(seq), [_window_stats(w) for w in _windows(seq)]))
            ii.append(index[seq])
        idx_a.append(ii[0])
        idx_b.append(ii[1])
        y.append(int(r.get("chose_left", 0)))

    S = len(table)
    keys = ["L", "k", "lm", "lb"] + [f"{c}{p}" for p in PERIODS for c in ("c", "m", "f")]
    arr = {key: np.zeros((S, MAX_WINDOWS), dtype="float64") for key in keys}
    mask = np.zeros((S, MAX_WINDOWS), dtype="float64")
    for s, (_, wins) in enumerate(table):
        for j, st in enumerate(wins):
            for key in keys:
                arr[key][s, j] = st[key]
            mask[s, j] = 1.0
        for j in range(len(wins), MAX_WINDOWS):  # harmless padding values
            arr["L"][s, j] = 3.0
            for p in PERIODS:
                arr[f"f{p}"][s, j] = float(min(p, 3))
    out = {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "seq_len": np.asarray([t[0] for t in table], dtype="float64"),
        "win_mask": mask,
    }
    for key in keys:
        out[f"win_{key}"] = arr[key]
    return out


def _zeros2():
    return np.zeros((1, MAX_WINDOWS), dtype="float64")


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    seq_len = pm.Data("seq_len", np.full(1, 8.0))
    win_mask = pm.Data("win_mask", _zeros2())
    win_L = pm.Data("win_L", np.full((1, MAX_WINDOWS), 3.0))
    win_k = pm.Data("win_k", _zeros2())
    win_lm = pm.Data("win_lm", _zeros2())
    win_lb = pm.Data("win_lb", _zeros2())
    win_c = {p: pm.Data(f"win_c{p}", _zeros2()) for p in PERIODS}
    win_m = {p: pm.Data(f"win_m{p}", _zeros2()) for p in PERIODS}
    win_f = {p: pm.Data(f"win_f{p}", np.ones((1, MAX_WINDOWS))) for p in PERIODS}

    # Slip rate of the repeating-motif generator.
    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    eps = pm.math.sigmoid(logit_eps)
    log_eps = pt.log(eps)
    log_1m_eps = pt.log1p(-eps)

    motif_terms = [
        -win_f[p] * math.log(2.0) + (win_c[p] - win_m[p]) * log_1m_eps + win_m[p] * log_eps
        for p in PERIODS
    ]
    log_motif = pm.math.logsumexp(pt.stack(motif_terms, axis=0), axis=0, keepdims=False) - math.log(len(PERIODS))
    # Log evidence of each window under the regular generators (equal weights).
    log_regular = pm.math.logsumexp(
        pt.stack([win_lm, win_lb, log_motif], axis=0), axis=0, keepdims=False
    ) - math.log(3.0)

    # Believed switch probability of a fair coin (shared).
    logit_q = pm.Normal("logit_q", mu=0.0, sigma=1.0)
    log_q = -pt.softplus(-logit_q)
    log_1mq = -pt.softplus(logit_q)

    # Worst-stretch evidence computed once per distinct sequence (S x windows).
    log_chance = math.log(0.5) + win_k * log_q + (win_L - 1.0 - win_k) * log_1mq
    lbf = pt.where(win_mask > 0.5, log_regular - log_chance, -1e3)
    # Soft maximum in nats: dominated by the single most damning stretch
    # (smooth, so NUTS is not faced with a kink wherever the argmax flips).
    worst = pm.math.logsumexp(lbf, axis=1, keepdims=False)

    worst_diff = worst[idx_a] - worst[idx_b]

    # Shared decisiveness.
    beta = pm.LogNormal("beta", mu=0.0, sigma=1.0)

    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    length_scale = pt.exp(-gamma * pt.log(seq_len[idx_a] / 5.0))

    eta = -beta * worst_diff * length_scale
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
