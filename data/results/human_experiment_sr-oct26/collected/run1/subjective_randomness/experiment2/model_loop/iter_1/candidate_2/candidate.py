"""Running-lead surprise along the tally path.

People keep a running tally of heads minus tails as they read a sequence left
to right and, at every flip, judge how surprising the current lead is for a fair
coin given the number of flips seen so far (lead^2 / flips, the prefix's
chi-square). A sequence's randomness impression is the average of this
moment-by-moment surprise over the path, compared (on a log scale) with each
person's own ideal level of surprise; an opening streak therefore counts against
a sequence far more than the same streak at the end. On some trials a person
guesses at a personal lapse rate.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def log_path_surprise(seq):
        seq = seq.strip().upper()
        lead = 0
        total = 0.0
        for t, c in enumerate(seq, start=1):
            lead += 1 if c == "H" else -1
            total += lead * lead / t
        return math.log(total / len(seq))

    return {
        "log_surprise_a": log_path_surprise(sequence_a),
        "log_surprise_b": log_path_surprise(sequence_b),
    }


with pm.Model() as model:
    log_surprise_a = pm.Data("log_surprise_a", np.zeros(1, dtype="float64"))
    log_surprise_b = pm.Data("log_surprise_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal (log) path surprise; a fair coin's expectation is log(1) = 0.
    mu_ideal = pm.Normal("mu_ideal", mu=0.0, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", mu_ideal + sigma_ideal * z_ideal)

    # Shared sensitivity to distance from the ideal.
    beta = pm.LogNormal("beta", mu=0.5, sigma=0.75)

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    c = ideal[participant_id]
    score_a = -beta * pt.sqr(log_surprise_a - c)
    score_b = -beta * pt.sqr(log_surprise_b - c)
    lam = lapse[participant_id]
    p_engaged = pm.math.sigmoid(score_a - score_b)
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
