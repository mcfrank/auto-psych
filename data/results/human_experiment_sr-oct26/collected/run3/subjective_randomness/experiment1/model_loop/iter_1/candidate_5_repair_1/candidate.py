"""Goldilocks gambler's-fallacy surprise (refinement of recency_weighted_gamblers_surprise).

People read each sequence flip by flip, predicting the next flip with a
gambler's-fallacy expectation (the longer the current run, the more they
expect it to break), with late flips weighing more than early ones. A
sequence looks random when this recency-weighted surprise is close to a
moderate "just-right" level, not when it is minimal: too much surprise (long
runs) and too little (perfect alternation, confirming every expected reversal)
both look non-random. The single change from the parent model is this concave
(quadratic, peaked) scoring of the felt surprise: one curvature parameter on
the parent's fixed decision scale, so no free sensitivity trades off against
the size of the surprise.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_T = 7  # transitions in a length-8 sequence
N_SLOTS = 400
CENTER = -0.7  # centring of the mean log predictive probability per flip


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

    def surprise(rep, run, pos, mask):
        logit_rep = a - b * (run - 1.0)
        logp = rep * -pt.softplus(-logit_rep) + (1.0 - rep) * -pt.softplus(logit_rep)
        w = mask * pt.exp(lam * pos)
        return pt.sum(w * logp, axis=1) / pt.sum(w, axis=1) - CENTER

    s_a = surprise(*seqs["a"])
    s_b = surprise(*seqs["b"])

    # Concave score on the parent's fixed scale (8, as in the parent model, so
    # the gambler's belief a, b alone sets the size of the felt surprise):
    # 8 * (s - kappa * s^2). kappa = 0 is the parent; kappa > 0 puts the most
    # random-looking surprise at a "just-right" level CENTER + 1 / (2 kappa).
    kappa = pm.HalfNormal("kappa", sigma=2.0)
    diff = 8.0 * ((s_a - s_b) - kappa * (s_a**2 - s_b**2))

    # Per-person sensitivity multiplier (log-normal, mean scale fixed at 1;
    # non-centred, spare slots for new participants).
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5)
    z = pm.Normal("z", 0.0, 1.0, shape=N_SLOTS)
    beta = pt.exp(sigma_beta * z)

    p_left = pm.Deterministic(
        "p_left",
        pt.clip(pm.math.sigmoid(beta[participant_id] * diff), 1e-6, 1 - 1e-6),
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
