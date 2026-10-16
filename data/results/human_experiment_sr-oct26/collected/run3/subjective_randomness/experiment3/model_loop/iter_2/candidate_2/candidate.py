"""People judge randomness by how compressible a sequence is when read once
left to right: they parse it into Lempel-Ziv phrases (each the shortest stretch
not already seen earlier), and a sequence that keeps producing new phrases
looks random while one that repeats what came before looks designed. The phrase
count is standardised against a fair coin's distribution at that length; the
sequence with the higher standardised count is chosen, with person-specific
decisiveness and side habit."""
import itertools
from functools import lru_cache

import numpy as np
import pymc as pm


def _lz76(seq):
    """Kaspar-Schuster LZ76 complexity (number of phrases)."""
    n = len(seq)
    if n <= 1:
        return n
    c, l, i, k, k_max = 1, 1, 0, 1, 1
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
    return c


@lru_cache(maxsize=None)
def _fair_coin_stats(n):
    vals = np.array([_lz76("".join(s)) for s in itertools.product("HT", repeat=n)], dtype=float)
    sd = vals.std()
    return vals.mean(), (sd if sd > 0 else 1.0)


def _standardised(seq):
    seq = seq.strip().upper()
    mu, sd = _fair_coin_stats(len(seq))
    return (_lz76(seq) - mu) / sd


def compute_features(sequence_a, sequence_b):
    return {"lz_diff": float(_standardised(sequence_a) - _standardised(sequence_b))}


N_SLOTS = 400

with pm.Model() as model:
    lz_diff = pm.Data("lz_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", 0.0, 1.0, shape=N_SLOTS)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z_beta)

    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.5)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.5)
    z_side = pm.Normal("z_side", 0.0, 1.0, shape=N_SLOTS)
    side = mu_side + sigma_side * z_side

    eta = side[participant_id] + beta[participant_id] * lz_diff
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
