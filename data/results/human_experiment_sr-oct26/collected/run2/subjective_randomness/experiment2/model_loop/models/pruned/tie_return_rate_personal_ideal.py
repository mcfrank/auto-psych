"""Tie-return rate personal ideal.

People read a sequence flip by flip while tracking whether heads and tails are
currently even, and judge randomness by how often the running count returns to
a tie, as a share of the most ties a sequence of that length could reach. Each
person expects a fair coin to restore balance at their own typical rate; a
sequence that rarely gets back to even (streaks, lopsided counts) or that ties
every two flips (too self-correcting) looks non-random, and people choose the
sequence whose tie-return rate is closer to their own expectation. Each person
also has a small left/right response bias.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def tie_share(seq):
        tally, ties = 0, 0
        for c in seq:
            tally += 1 if c == "H" else -1
            if tally == 0:
                ties += 1
        return ties / (len(seq) // 2)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "tie_a": tie_share(a),
        "tie_b": tie_share(b),
        "log_len": float(np.log(len(a) / 6.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    tie_a = pm.Data("tie_a", np.zeros(1, dtype="float64"))
    tie_b = pm.Data("tie_b", np.zeros(1, dtype="float64"))
    log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific expected tie-return rate, in (0, 1).
    mu_ideal = pm.Normal("mu_ideal", mu=0.0, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Person-specific sensitivity, scaled by a power of the length.
    mu_log_beta = pm.Normal("mu_log_beta", mu=1.5, sigma=1.5)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z_beta)
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)

    # Person-specific left/right response bias (decision stage, not a cue).
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    theta = ideal[participant_id]
    dist_a = (tie_a - theta) ** 2
    dist_b = (tie_b - theta) ** 2
    sens = beta[participant_id] * pm.math.exp(lam * log_len)
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(sens * (dist_b - dist_a) + side[participant_id])
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
