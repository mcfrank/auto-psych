"""Gist typicality: a sequence looks random to the extent that many fair-coin
sequences of its length share its gist -- its number of heads and its number of
runs. Rare gists (lopsided counts, long streaks, strict alternation) look
non-random; people differ only in how strongly this typicality drives choice."""
import functools
import itertools
import math

import numpy as np
import pymc as pm


@functools.lru_cache(maxsize=None)
def _gist_counts(n):
    counts = {}
    for seq in itertools.product("HT", repeat=n):
        k = seq.count("H")
        r = 1 + sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        counts[(k, r)] = counts.get((k, r), 0) + 1
    return counts


def _log_typicality(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = seq.count("H")
    r = 1 + sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    return math.log(_gist_counts(n)[(k, r)])


def compute_features(sequence_a, sequence_b):
    return {"typ_diff": _log_typicality(sequence_a) - _log_typicality(sequence_b)}


with pm.Model() as model:
    typ_diff = pm.Data("typ_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific sensitivity to gist typicality (non-centred, log scale).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z = pm.Normal("z", mu=0.0, sigma=1.0, shape=400)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z)

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta[participant_id] * typ_diff)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
