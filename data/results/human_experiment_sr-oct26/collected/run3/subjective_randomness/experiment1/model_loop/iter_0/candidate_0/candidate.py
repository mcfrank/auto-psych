"""Negative-recency expectation fit.

People read each sequence flip by flip and check every flip against a
gambler's-fallacy expectation: after a streak of length r they expect a
switch with probability sigmoid(a + b*(r-1)), rising with streak length.
A sequence looks random to the extent its flips conform to that expectation
(summed log-probability of its transitions under the expectation), with
later-read flips weighted more (recency), and people differ in how strongly
this conformity drives choice (person-level sensitivity).
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_T = 7  # transitions in a length-8 sequence
N_SLOTS = 400


def _seq_id(seq):
    """Index of an H/T sequence of length 2-8 in the table below."""
    seq = seq.strip().upper()
    code = int("".join("1" if c == "H" else "0" for c in seq), 2)
    return (2 ** len(seq) - 4) + code


def _transitions(seq):
    alt, run, dist, mask = [], [], [], []
    r = 1
    n = len(seq) - 1
    for k in range(1, len(seq)):
        a = 1.0 if seq[k] != seq[k - 1] else 0.0
        alt.append(a)
        run.append(float(r))
        dist.append(float(n - k))  # 0 for the last flip
        mask.append(1.0)
        r = 1 if a else r + 1
    pad = MAX_T - len(alt)
    return alt + [0.0] * pad, run + [1.0] * pad, dist + [0.0] * pad, mask + [0.0] * pad


# Constant table: one row per possible sequence of length 2-8.
_rows = []
for _L in range(2, 9):
    for _bits in itertools.product("TH", repeat=_L):
        _s = "".join(_bits)
        assert _seq_id(_s) == len(_rows)
        _rows.append(_transitions(_s))
_TABLE = np.array(_rows, dtype="float64")  # (n_seq, 4, MAX_T)
ALT, RUN, DIST, MASK = (_TABLE[:, i, :] for i in range(4))


def compute_features(sequence_a, sequence_b):
    return {"seq_id_a": _seq_id(sequence_a), "seq_id_b": _seq_id(sequence_b)}


with pm.Model() as model:
    seq_id_a = pm.Data("seq_id_a", np.zeros(1, dtype="int64"))
    seq_id_b = pm.Data("seq_id_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Expected switch probability after a streak of length r: sigmoid(a + b*(r-1)).
    a = pm.Normal("a", mu=0.0, sigma=1.0)
    b = pm.HalfNormal("b", sigma=1.0)
    # Recency: weight of a flip d positions before the end is exp(-gamma*d).
    gamma = pm.HalfNormal("gamma", sigma=0.5)

    # Person-level sensitivity (non-centred, log scale).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z = pm.Normal("z", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z)

    # Conformity score of every possible sequence, then gathered per trial.
    logit_q = a + b * (RUN - 1.0)
    logp = ALT * (-pt.softplus(-logit_q)) + (1.0 - ALT) * (-pt.softplus(logit_q))
    w = pt.exp(-gamma * DIST) * MASK
    score = pt.sum(w * logp, axis=1)

    diff = score[seq_id_a] - score[seq_id_b]
    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(beta[participant_id] * diff), 1e-6, 1 - 1e-6)
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
