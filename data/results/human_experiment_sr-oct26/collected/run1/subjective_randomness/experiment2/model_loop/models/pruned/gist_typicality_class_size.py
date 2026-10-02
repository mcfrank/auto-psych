"""Gist typicality: people register only a sequence's number of heads and its
number of runs, and judge it random to the extent that many sequences of the
same length share that gist (log size of its equivalence class). The sequence
with the more common gist is chosen, with a personal decisiveness, and people
guess on some trials at a personal lapse rate."""

import math

import numpy as np
import pymc as pm


def _log_class_size(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = seq.count("H")
    m = n - k
    r = 1 + sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    if k == 0 or m == 0:
        return 0.0

    def c(a, b):
        return math.comb(a, b) if 0 <= b <= a else 0

    if r % 2 == 0:
        h = r // 2
        count = 2 * c(k - 1, h - 1) * c(m - 1, h - 1)
    else:
        big, small = (r + 1) // 2, (r - 1) // 2
        count = c(k - 1, big - 1) * c(m - 1, small - 1) + c(k - 1, small - 1) * c(m - 1, big - 1)
    return math.log(max(count, 1))


def compute_features(sequence_a, sequence_b):
    return {"gist_diff": _log_class_size(sequence_a) - _log_class_size(sequence_b)}


N_PEOPLE = 400

with pm.Model() as model:
    gist_diff = pm.Data("gist_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", 0.0, 1.0, shape=N_PEOPLE)
    beta = pm.math.exp(mu_beta + sigma_beta * z_beta)

    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", 0.0, 1.0, shape=N_PEOPLE)
    lapse = pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse)

    b = beta[participant_id]
    lp = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", 0.5 * lp + (1.0 - lp) * pm.math.sigmoid(b * gist_diff)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
