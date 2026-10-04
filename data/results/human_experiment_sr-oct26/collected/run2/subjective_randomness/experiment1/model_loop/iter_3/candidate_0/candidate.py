"""Running-tally drift ideal.

People read a sequence flip by flip while keeping a running tally of how far
heads lead tails, and judge randomness by how far that tally wanders from
balance along the way: each person expects a fair coin's tally to drift from
even by some typical amount (scaled to the number of flips read so far), and a
sequence looks random to the extent that its average drift matches that
personal expectation. Long streaks make the tally run away (too much drift);
strict alternation pins it at even (too little drift).
"""
import math

import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def tally_drift(seq):
        seq = seq.strip().upper()
        lead = 0
        total = 0.0
        for k, c in enumerate(seq, start=1):
            lead += 1 if c == "H" else -1
            total += abs(lead) / math.sqrt(k)
        return total / len(seq)

    return {"drift_a": tally_drift(sequence_a), "drift_b": tally_drift(sequence_b)}


N_SLOTS = 400

with pm.Model() as model:
    drift_a = pm.Data("drift_a", np.zeros(1, dtype="float64"))
    drift_b = pm.Data("drift_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of each person's expected tally drift (log scale).
    mu_ideal = pm.Normal("mu_ideal", mu=math.log(0.7), sigma=0.5)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=0.5)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.exp(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to squared mismatch between observed and expected drift.
    beta = pm.HalfNormal("beta", sigma=10.0)

    theta = ideal[participant_id]
    dist_a = (drift_a - theta) ** 2
    dist_b = (drift_b - theta) ** 2
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (dist_b - dist_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
