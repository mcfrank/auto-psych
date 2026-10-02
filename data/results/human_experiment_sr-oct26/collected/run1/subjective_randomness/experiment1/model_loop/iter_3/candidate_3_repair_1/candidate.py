"""Personal lapse rate on a locally applied personal-ideal-alternation judgment.

Refinement of `personal_lapse_ideal_alternation`: people judge a sequence as
more random the closer its switching lies to their own personal ideal
switching rate, and guess on some trials at a personal lapse rate. The single
change: the ideal is applied locally — every sliding stretch of three
consecutive transitions is expected to switch at the ideal rate, and a
sequence's non-randomness is the average squared departure of its local
switching rates from the ideal. Bunching repeats into one streak (and switches
into a rigid alternating stretch) is then penalised even when the overall
alternation rate is unchanged.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
WINDOW_TRANSITIONS = 3


def compute_features(sequence_a, sequence_b):
    def local_rates(seq):
        seq = seq.strip().upper()
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        switches = [1.0 if x != y else 0.0 for x, y in zip(seq, seq[1:])]
        k = min(WINDOW_TRANSITIONS, len(switches))
        return np.array(
            [np.mean(switches[i : i + k]) for i in range(len(switches) - k + 1)]
        )

    ra = local_rates(sequence_a)
    rb = local_rates(sequence_b)
    # mean over windows of (r - ideal)^2 = mean(r^2) - 2 ideal mean(r) + ideal^2
    return {
        "loc_mean_a": float(ra.mean()),
        "loc_sq_a": float(np.mean(ra**2)),
        "loc_mean_b": float(rb.mean()),
        "loc_sq_b": float(np.mean(rb**2)),
    }


with pm.Model() as model:
    loc_mean_a = pm.Data("loc_mean_a", np.zeros(1, dtype="float64"))
    loc_sq_a = pm.Data("loc_sq_a", np.zeros(1, dtype="float64"))
    loc_mean_b = pm.Data("loc_mean_b", np.zeros(1, dtype="float64"))
    loc_sq_b = pm.Data("loc_sq_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal switching rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    beta = pm.LogNormal("beta", mu=2.0, sigma=0.5)

    # Personal lapse (guessing) rate (non-centred logit-normal population).
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    # Average squared local departure from the ideal (non-randomness).
    dev_a = loc_sq_a - 2.0 * theta * loc_mean_a + pt.sqr(theta)
    dev_b = loc_sq_b - 2.0 * theta * loc_mean_b + pt.sqr(theta)
    p_engaged = pm.math.sigmoid(beta * (dev_b - dev_a))
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
