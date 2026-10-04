"""Lempel-Ziv copy complexity.

People judge randomness by how hard a sequence is to describe by copying:
reading left to right they chunk it into the fewest pieces such that each new
piece extends something already seen earlier in the sequence by one novel flip
(Lempel-Ziv 1976 parsing). More pieces = less compressible = more random.
Streaks, strict alternation and repeated units are cheap to describe and look
non-random. People differ in how strongly this felt incompressibility drives
their choice, and each has a small habitual left/right lean.
"""
import math

import numpy as np
import pymc as pm


def _lz76(seq):
    """Number of phrases in the Lempel-Ziv (1976) parse (Kaspar-Schuster)."""
    n = len(seq)
    if n <= 1:
        return float(n)
    c, i, k, l, k_max = 1, 0, 1, 1, 1
    while True:
        if seq[i + k - 1] == seq[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return float(c)


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    n = len(a)
    norm = math.log2(n) / n if n > 1 else 1.0
    return {
        "lz_diff": (_lz76(a) - _lz76(b)) * norm,
        "log_len": float(np.log(n / 6.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    lz_diff = pm.Data("lz_diff", np.zeros(1, dtype="float64"))
    log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific sensitivity to incompressibility (sign free).
    mu_beta = pm.Normal("mu_beta", mu=1.0, sigma=2.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = mu_beta + sigma_beta * z_beta
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)

    # Person-specific left/right response lean.
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    logit = beta[participant_id] * pm.math.exp(lam * log_len) * lz_diff + side[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
