"""Goldilocks gambler's-fallacy surprise (refinement of recency_weighted_gamblers_surprise).

People read each sequence flip by flip, predicting the next flip with a
gambler's-fallacy expectation (the longer the current run, the more they
expect it to break), with late flips weighing more than early ones. A
sequence looks random when this recency-weighted surprise is close to a
moderate "just-right" level, not when it is minimal: too much surprise (long
runs) and too little (perfect alternation, confirming every expected reversal)
both look non-random. The single change from the parent model is this
target-level (squared-distance) scoring of the felt surprise.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_T = 7  # transitions in a length-8 sequence
N_SLOTS = 400


def _transitions(seq):
    seq = seq.strip().upper()
    n = len(seq)
    rep, run, pos, mask = [], [], [], []
    k = 1
    for t in range(1, n):
        r = 1.0 if seq[t] == seq[t - 1] else 0.0
        rep.append(r)
        run.append(float(k))  # run length before this flip
        pos.append(t / (n - 1))
        mask.append(1.0)
        k = k + 1 if r else 1
    while len(rep) < MAX_T:
        rep.append(0.0)
        run.append(1.0)
        pos.append(0.0)
        mask.append(0.0)
    return rep, run, pos, mask


def compute_features(sequence_a, sequence_b):
    out = {}
    for side, seq in (("a", sequence_a), ("b", sequence_b)):
        rep, run, pos, mask = _transitions(seq)
        for i in range(MAX_T):
            out[f"rep_{side}_{i}"] = rep[i]
            out[f"run_{side}_{i}"] = run[i]
            out[f"pos_{side}_{i}"] = pos[i]
            out[f"mask_{side}_{i}"] = mask[i]
    return out


def _stack(side, name):
    cols = [pm.Data(f"{name}_{side}_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_T)]
    return pt.stack(cols, axis=1)


with pm.Model() as model:
    seqs = {}
    for side in ("a", "b"):
        seqs[side] = tuple(_stack(side, nm) for nm in ("rep", "run", "pos", "mask"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Gambler's-fallacy predictor: logit P(repeat | run length k) = a - b*(k-1)
    a = pm.Normal("a", mu=0.0, sigma=1.5)
    b = pm.HalfNormal("b", sigma=1.5)
    # Recency of surprise memory: weight exp(lam * position)
    lam = pm.Normal("lam", mu=0.0, sigma=1.5)
    # The "just-right" surprise level (mean log predictive probability per flip).
    s0 = pm.Normal("s0", mu=-0.7, sigma=0.5)

    def surprise(rep, run, pos, mask):
        logit_rep = a - b * (run - 1.0)
        logp = rep * -pt.softplus(-logit_rep) + (1.0 - rep) * -pt.softplus(logit_rep)
        w = mask * pt.exp(lam * pos)
        return pt.sum(w * logp, axis=1) / pt.sum(w, axis=1)

    score_a = -((surprise(*seqs["a"]) - s0) ** 2)
    score_b = -((surprise(*seqs["b"]) - s0) ** 2)

    # Per-person sensitivity (log-normal population, non-centred, spare slots).
    mu_log_beta = pm.Normal("mu_log_beta", mu=1.5, sigma=1.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5)
    z = pm.Normal("z", 0.0, 1.0, shape=N_SLOTS)
    beta = pt.exp(mu_log_beta + sigma_beta * z)

    p_left = pm.Deterministic(
        "p_left",
        pt.clip(pm.math.sigmoid(beta[participant_id] * (score_a - score_b)), 1e-6, 1 - 1e-6),
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
