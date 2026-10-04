"""Bayesian triplet-chunk rig detector.

People judge randomness like an ideal Bayesian detector working over three-flip
chunks: is the bag of a sequence's overlapping three-flip chunks better explained
by a fair coin (each of the eight chunks equally likely) or by a rigged process
favouring some chunks (an unknown chunk distribution, uniform Dirichlet prior)?
The distortion is a resource limit: the sequence is encoded only as a bag of
three-flip chunks, ignoring their overlap and position. How strongly this
evidence drives choice is shared; each person has a small left/right lean.
"""
import itertools
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

TRIPLETS = ["".join(t) for t in itertools.product("HT", repeat=3)]


def _log_odds_random(seq):
    """log P(chunks | fair coin) - log P(chunks | rigged chunk process)."""
    seq = seq.strip().upper()
    counts = {t: 0 for t in TRIPLETS}
    for i in range(len(seq) - 2):
        counts[seq[i:i + 3]] += 1
    m = sum(counts.values())
    log_fair = -m * math.log(8.0)
    # Dirichlet(1,...,1)-categorical marginal likelihood.
    log_rig = math.lgamma(8.0) - math.lgamma(8.0 + m) + sum(
        math.lgamma(1.0 + c) for c in counts.values()
    )
    return log_fair - log_rig


def compute_features(sequence_a, sequence_b):
    return {"chunk_evidence_diff": _log_odds_random(sequence_a) - _log_odds_random(sequence_b)}


with pm.Model() as model:
    evidence_diff = pm.Data("chunk_evidence_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Shared sensitivity to the fair-vs-rigged evidence.
    beta = pm.HalfNormal("beta", sigma=2.0)

    # Person-specific left/right lean (non-centred).
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.5)
    z_side = pm.Normal("z_side", 0.0, 1.0, shape=400)
    side = sigma_side * z_side

    eta = beta * evidence_diff + side[participant_id]
    p_left = pm.Deterministic("p_left", pt.clip(pm.math.sigmoid(eta), 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
