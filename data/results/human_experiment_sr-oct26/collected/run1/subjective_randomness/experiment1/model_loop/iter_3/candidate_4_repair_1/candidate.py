"""Glimpse-wise personal-ideal alternation with a personal lapse rate.

Refinement of `personal_lapse_ideal_alternation`: people compare alternation
with their own ideal switching rate and guess on a person-specific share of
trials, but, as the single change, they judge alternation locally: each
four-flip glimpse's switching rate is compared with the ideal, and a sequence's
non-randomness is the mean squared deviation over its glimpses. Streaks beside
alternating stretches are thus penalised even when the overall rate is ideal.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
GLIMPSE_TRANSITIONS = 3  # a four-flip working-memory glimpse


def compute_features(sequence_a, sequence_b):
    def glimpse_moments(seq):
        seq = seq.strip().upper()
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        switches = [1.0 if x != y else 0.0 for x, y in zip(seq, seq[1:])]
        k = min(GLIMPSE_TRANSITIONS, len(switches))
        rates = np.array(
            [np.mean(switches[i:i + k]) for i in range(len(switches) - k + 1)]
        )
        return float(rates.mean()), float(np.mean(rates ** 2))

    m_a, q_a = glimpse_moments(sequence_a)
    m_b, q_b = glimpse_moments(sequence_b)
    return {"glimpse_mean_a": m_a, "glimpse_sq_a": q_a,
            "glimpse_mean_b": m_b, "glimpse_sq_b": q_b}


with pm.Model() as model:
    glimpse_mean_a = pm.Data("glimpse_mean_a", np.zeros(1, dtype="float64"))
    glimpse_sq_a = pm.Data("glimpse_sq_a", np.zeros(1, dtype="float64"))
    glimpse_mean_b = pm.Data("glimpse_mean_b", np.zeros(1, dtype="float64"))
    glimpse_sq_b = pm.Data("glimpse_sq_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal alternation rate (non-centred logit-normal population).
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
    # Mean over glimpses of (glimpse rate - ideal)^2, from the glimpse moments.
    dev_a = glimpse_sq_a - 2.0 * theta * glimpse_mean_a + pt.sqr(theta)
    dev_b = glimpse_sq_b - 2.0 * theta * glimpse_mean_b + pt.sqr(theta)
    p_engaged = pm.math.sigmoid(beta * (dev_b - dev_a))
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
