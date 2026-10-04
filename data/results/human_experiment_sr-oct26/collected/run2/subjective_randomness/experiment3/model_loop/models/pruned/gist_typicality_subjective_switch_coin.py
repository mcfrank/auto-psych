"""Gist typicality under a personal subjective switching coin.

People encode a sequence only by its gist -- the span of its running
heads-minus-tails tally, its number of side switches and its number of distinct
three-flip chunks -- and judge it random to the extent that a random coin would
often produce a sequence with that gist (normative typicality). The one
distortion: the random coin they imagine switches sides at a personal rate q
(usually above one half), so a gist's probability is
    P(gist) = N(gist) * 0.5 * q^s * (1 - q)^(n - 1 - s),
with N(gist) the number of length-n sequences sharing the gist. People choose
the sequence with the larger log typicality, with a personal decisiveness and a
personal left/right lean.
"""
import itertools
import math

import numpy as np
import pymc as pm


def _gist(seq):
    tally, hi, lo = 0, 0, 0
    for c in seq:
        tally += 1 if c == "H" else -1
        hi = max(hi, tally)
        lo = min(lo, tally)
    switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    chunks = len({seq[i:i + 3] for i in range(len(seq) - 2)})
    return (hi - lo, switches, chunks)


_COUNTS = {}
for _n in range(2, 9):
    for _t in itertools.product("HT", repeat=_n):
        _g = (_n,) + _gist("".join(_t))
        _COUNTS[_g] = _COUNTS.get(_g, 0) + 1


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    ga = _gist(a)
    gb = _gist(b)
    log_n_a = math.log(_COUNTS[(len(a),) + ga])
    log_n_b = math.log(_COUNTS[(len(b),) + gb])
    return {
        "log_count_diff": log_n_a - log_n_b,
        "switch_diff": float(ga[1] - gb[1]),
    }


N_SLOTS = 400

with pm.Model() as model:
    log_count_diff = pm.Data("log_count_diff", np.zeros(1, dtype="float64"))
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal subjective switching rate of the imagined random coin (logit scale).
    mu_q = pm.Normal("mu_q", mu=0.3, sigma=1.0)
    sigma_q = pm.HalfNormal("sigma_q", sigma=0.7)
    z_q = pm.Normal("z_q", mu=0.0, sigma=1.0, shape=N_SLOTS)
    logit_q = mu_q + sigma_q * z_q

    # Personal decisiveness on the log-typicality difference.
    mu_log_beta = pm.Normal("mu_log_beta", mu=-0.5, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z_beta)

    # Personal left/right lean.
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    # Equal lengths: log q^s (1-q)^(n-1-s) difference = switch_diff * logit(q).
    typ_diff = log_count_diff + switch_diff * logit_q[participant_id]
    logit = beta[participant_id] * typ_diff + side[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
