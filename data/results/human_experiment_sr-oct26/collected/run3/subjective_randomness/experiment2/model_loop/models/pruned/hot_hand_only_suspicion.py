"""Hot-hand-only suspicion: Bayesian chance versus a streaky or a biased coin.

People judge a sequence random to the extent a plain fair coin explains it
better than the only non-random coins they suspect: a hot-hand (streaky) coin
that repeats its last outcome with an unknown probability r (prior Beta(alpha, 1),
alpha fitted, so suspicion is of repeating, never of switching), or a coin with
an unknown heads bias (uniform prior). Both unknowns are integrated out. Switching
is never suspicious, so perfect alternation and periodic motifs look maximally
random — where this disagrees most with the incumbent, which condemns them as
repeating motifs. The evidence is weighed per flip (divided by a fitted power of
the length) and people differ in decision sensitivity.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _summary(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    log_biased = math.lgamma(h + 1) + math.lgamma(n - h + 1) - math.lgamma(n + 2)
    return {"n": float(n), "k": float(k), "rep": float(n - 1 - k), "lb": log_biased}


def prepare_observed(rows):
    """Table of distinct sequences plus per-trial indices into it."""
    index, table = {}, []
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
    return {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "seq_len": np.asarray([t["n"] for t in table], dtype="float64"),
        "seq_switch": np.asarray([t["k"] for t in table], dtype="float64"),
        "seq_repeat": np.asarray([t["rep"] for t in table], dtype="float64"),
        "seq_lbias": np.asarray([t["lb"] for t in table], dtype="float64"),
    }


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_switch = pm.Data("seq_switch", np.zeros(1, dtype="float64"))
    seq_repeat = pm.Data("seq_repeat", np.ones(1, dtype="float64"))
    seq_lbias = pm.Data("seq_lbias", np.zeros(1, dtype="float64"))

    # Streakiness people suspect: repeat probability r ~ Beta(alpha, 1), alpha > 1.
    log_alpha_m1 = pm.Normal("log_alpha_m1", mu=0.0, sigma=1.0)
    alpha = pm.Deterministic("alpha", 1.0 + pt.exp(log_alpha_m1))

    # Log marginal likelihood under the hot-hand coin (first flip free).
    log_streaky = (
        -math.log(2.0)
        + pt.gammaln(seq_repeat + alpha) + pt.gammaln(seq_switch + 1.0)
        - pt.gammaln(seq_repeat + seq_switch + alpha + 1.0)
        - (pt.gammaln(alpha) - pt.gammaln(alpha + 1.0))
    )
    log_regular = pm.math.logsumexp(
        pt.stack([log_streaky, seq_lbias], axis=0), axis=0, keepdims=False
    ) - math.log(2.0)
    randomness = -seq_len * math.log(2.0) - log_regular

    # Per-flip weighing of the evidence.
    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    length_scale = pt.exp(-gamma * pt.log(seq_len[idx_a] / 5.0))
    score_diff = (randomness[idx_a] - randomness[idx_b]) * length_scale

    # Person-specific decision sensitivity (non-centred log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta[participant_id] * score_diff))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
