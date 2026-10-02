"""Worst-stretch switch deficit.

People judge a coin-flip sequence by its single most un-random-looking stretch:
over every contiguous stretch of flips they count how far its number of switches
falls short of (a streak) or exceeds (rigid alternation) the number their own
ideal switching rate predicts, and the sequence is only as random as that worst
stretch. The deficit is counted in switches, so a long modestly-off stretch can
outweigh a short badly-off one. People also guess on some trials at a personal
lapse rate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
MAX_TRANS = 7  # sequences of up to 8 flips

# Every contiguous stretch of transitions (start i, end j exclusive).
_WINDOWS = [(i, j) for i in range(MAX_TRANS) for j in range(i + 1, MAX_TRANS + 1)]
_W = np.zeros((len(_WINDOWS), MAX_TRANS))
for _k, (_i, _j) in enumerate(_WINDOWS):
    _W[_k, _i:_j] = 1.0
_WIN_LEN = _W.sum(axis=1)
_WIN_END = np.array([j for _, j in _WINDOWS], dtype="float64")


def compute_features(sequence_a, sequence_b):
    def feats(seq, tag):
        seq = seq.strip().upper()
        if not 2 <= len(seq) <= MAX_TRANS + 1:
            raise ValueError(f"sequence length out of range: {seq!r}")
        t = [1.0 if x != y else 0.0 for x, y in zip(seq, seq[1:])]
        t += [0.0] * (MAX_TRANS - len(t))
        out = {f"sw{k}_{tag}": t[k] for k in range(MAX_TRANS)}
        out[f"ntrans_{tag}"] = float(len(seq) - 1)
        return out

    return {**feats(sequence_a, "a"), **feats(sequence_b, "b")}


def worst_stretch(sw, ntrans, ideal):
    """Largest squared switch-count deviation over stretches, per trial (/ n^2)."""
    counts = pt.dot(sw, _W.T)  # trials x windows
    expected = ideal[:, None] * _WIN_LEN[None, :]
    valid = pt.le(_WIN_END[None, :], ntrans[:, None])
    dev = pt.switch(valid, pt.sqr(counts - expected), 0.0)
    return pt.max(dev, axis=1) / pt.sqr(ntrans)


with pm.Model() as model:
    sw_a = pt.stack(
        [pm.Data(f"sw{k}_a", np.zeros(1, dtype="float64")) for k in range(MAX_TRANS)], axis=1
    )
    sw_b = pt.stack(
        [pm.Data(f"sw{k}_b", np.zeros(1, dtype="float64")) for k in range(MAX_TRANS)], axis=1
    )
    ntrans_a = pm.Data("ntrans_a", np.ones(1, dtype="float64"))
    ntrans_b = pm.Data("ntrans_b", np.ones(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal switching rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.6, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to the worst stretch's deviation.
    beta = pm.LogNormal("beta", mu=2.0, sigma=0.7)

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    worst_a = worst_stretch(sw_a, ntrans_a, theta)
    worst_b = worst_stretch(sw_b, ntrans_b, theta)
    p_engaged = pm.math.sigmoid(beta * (worst_b - worst_a))
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
