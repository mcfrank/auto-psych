"""Leaky-memory Bayesian randomness judgment.

People judge randomness as Bayesian model comparison: a sequence looks random to
the extent a fair coin explains it better than the suspected non-random
generators (a biased coin with unknown bias, or a Markov coin with unknown
switching probability; both with uniform priors, equally likely a priori). The
one distortion is leaky memory: each flip's evidence is discounted
geometrically with its distance from the end of the sequence (rate `rho`), so
patterns at the end of a sequence weigh more than the same patterns at its
start. People also guess on some trials at a personal lapse rate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_LEN = 8
MAX_PARTICIPANTS = 400
LOG_HALF = float(np.log(0.5))


def _encode(seq):
    """Per-position indicators, indexed by distance k from the end (k = 0 last)."""
    seq = seq.strip().upper()
    n = len(seq)
    if n < 2 or n > MAX_LEN or set(seq) - {"H", "T"}:
        raise ValueError(f"invalid sequence: {seq!r}")
    valid = np.zeros(MAX_LEN)
    head = np.zeros(MAX_LEN)
    trans = np.zeros(MAX_LEN)
    rep = np.zeros(MAX_LEN)
    for k in range(n):
        i = n - 1 - k
        valid[k] = 1.0
        head[k] = 1.0 if seq[i] == "H" else 0.0
        if i >= 1:
            trans[k] = 1.0
            rep[k] = 1.0 if seq[i] == seq[i - 1] else 0.0
    return valid, head, trans, rep


def prepare_observed(rows):
    table = {}
    idx_a, idx_b = [], []
    for r in rows:
        for key, out in (("sequence_a", idx_a), ("sequence_b", idx_b)):
            s = str(r[key]).strip().upper()
            if s not in table:
                table[s] = len(table)
            out.append(table[s])
    seqs = sorted(table, key=table.get)
    enc = [_encode(s) for s in seqs]
    return {
        "seq_valid": np.array([e[0] for e in enc], dtype="float64"),
        "seq_head": np.array([e[1] for e in enc], dtype="float64"),
        "seq_trans": np.array([e[2] for e in enc], dtype="float64"),
        "seq_rep": np.array([e[3] for e in enc], dtype="float64"),
        "idx_a": np.array(idx_a, dtype="int64"),
        "idx_b": np.array(idx_b, dtype="int64"),
        "participant_id": np.array([int(r["participant_id"]) for r in rows], dtype="int64"),
        "chose_left": np.array([int(float(r.get("chose_left", 0))) for r in rows], dtype="int64"),
    }


def _lbeta(x, y):
    return pt.gammaln(x) + pt.gammaln(y) - pt.gammaln(x + y)


with pm.Model() as model:
    seq_valid = pm.Data("seq_valid", np.zeros((1, MAX_LEN), dtype="float64"))
    seq_head = pm.Data("seq_head", np.zeros((1, MAX_LEN), dtype="float64"))
    seq_trans = pm.Data("seq_trans", np.zeros((1, MAX_LEN), dtype="float64"))
    seq_rep = pm.Data("seq_rep", np.zeros((1, MAX_LEN), dtype="float64"))
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Memory retention per flip step back from the end (1 = no leak).
    rho = pm.Beta("rho", alpha=4.0, beta=2.0)
    w = rho ** pt.arange(MAX_LEN, dtype="float64")[None, :]

    # Discounted evidence counts per distinct sequence.
    wv = seq_valid * w
    n_eff = pt.sum(wv, axis=1)
    h_eff = pt.sum(wv * seq_head, axis=1)
    t_eff = n_eff - h_eff
    tw = seq_trans * w
    r_eff = pt.sum(tw * seq_rep, axis=1)
    s_eff = pt.sum(tw, axis=1) - r_eff
    w_first = pt.sum((seq_valid - seq_trans) * w, axis=1)

    # Log Bayes factor: fair coin vs. {biased coin, Markov coin} (uniform priors).
    log_fair = n_eff * LOG_HALF
    log_biased = _lbeta(h_eff + 1.0, t_eff + 1.0)
    log_markov = w_first * LOG_HALF + _lbeta(r_eff + 1.0, s_eff + 1.0)
    randomness = log_fair - (pt.logaddexp(log_biased, log_markov) + LOG_HALF)

    beta = pm.LogNormal("beta", mu=0.5, sigma=1.0)
    p_engaged = pm.math.sigmoid(beta * (randomness[idx_a] - randomness[idx_b]))

    mu_lapse = pm.Normal("mu_lapse", mu=-1.5, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))
    lam = lapse[participant_id]

    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
