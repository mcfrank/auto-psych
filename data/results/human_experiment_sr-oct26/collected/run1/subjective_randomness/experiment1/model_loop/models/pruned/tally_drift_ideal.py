"""Running-tally drift judged against a personal ideal.

People keep a running tally of heads minus tails as they read a sequence and
judge it random to the extent that the tally wanders from balance by about as
much as they expect a fair coin's tally to wander. The path matters, not only
the final count: a streak that drives the tally far from zero counts against a
sequence even if later corrected, and a tally that never leaves zero does too.
Each person has their own ideal amount of wandering and picks the sequence
whose tally drift is closer to it.

Drift = mean squared running lead, relative to its fair-coin expectation
((n + 1) / 2), on a log scale.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def log_drift(seq):
        seq = seq.strip().upper()
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        lead = 0
        total = 0.0
        for c in seq:
            lead += 1 if c == "H" else -1
            total += lead * lead
        n = len(seq)
        return math.log((total / n) / ((n + 1) / 2.0))

    return {"log_drift_a": log_drift(sequence_a), "log_drift_b": log_drift(sequence_b)}


with pm.Model() as model:
    log_drift_a = pm.Data("log_drift_a", np.zeros(1, dtype="float64"))
    log_drift_b = pm.Data("log_drift_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal log-drift (0 = a fair coin's expected wandering), non-centred.
    mu_ideal = pm.Normal("mu_ideal", mu=0.0, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", mu_ideal + sigma_ideal * z_ideal)
    beta = pm.LogNormal("beta", mu=0.0, sigma=1.0)

    theta = ideal[participant_id]
    score_a = -pt.sqr(log_drift_a - theta)
    score_b = -pt.sqr(log_drift_b - theta)
    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(beta * (score_a - score_b)), 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
