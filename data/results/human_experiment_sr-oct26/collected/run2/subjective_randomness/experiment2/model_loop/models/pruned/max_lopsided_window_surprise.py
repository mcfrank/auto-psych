"""People judge a sequence by its single most lopsided stretch: of every contiguous
stretch of flips, the one whose heads/tails split is most improbable for a fair
coin of that length decides how non-random the sequence looks (a maximum over
windows, not an average); people pick the sequence whose most lopsided stretch is
less surprising, differing in how strongly that surprise drives their choice.
"""
import math

import numpy as np
import pymc as pm


def _window_surprise(length, heads):
    """-log2 of the two-sided fair-coin probability of a split at least this lopsided."""
    dev = abs(heads - length / 2.0)
    total = sum(math.comb(length, k) for k in range(length + 1)
                if abs(k - length / 2.0) >= dev - 1e-9)
    return -math.log2(total / 2.0 ** length)


def _max_window_surprise(seq):
    seq = seq.strip().upper()
    n = len(seq)
    best = 0.0
    for i in range(n):
        heads = 0
        for j in range(i, n):
            heads += seq[j] == "H"
            length = j - i + 1
            if length >= 2:
                best = max(best, _window_surprise(length, heads))
    return best


def compute_features(sequence_a, sequence_b):
    return {
        "max_surprise_a": _max_window_surprise(sequence_a),
        "max_surprise_b": _max_window_surprise(sequence_b),
    }


with pm.Model() as model:
    max_surprise_a = pm.Data("max_surprise_a", np.zeros(1, dtype="float64"))
    max_surprise_b = pm.Data("max_surprise_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific sensitivity to the most surprising stretch (non-centred).
    mu_beta = pm.Normal("mu_beta", mu=0.5, sigma=1.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = mu_beta + sigma_beta * z_beta

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta[participant_id] * (max_surprise_b - max_surprise_a)),
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
