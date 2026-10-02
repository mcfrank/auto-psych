"""Personal lapse rate on a personal-ideal-alternation judgment.

People judge a sequence as more random the closer its proportion of
alternations lies to their own personal ideal switching rate, but on some
trials the decision rule is not engaged and the person guesses. The guessing
(lapse) rate is a stable per-person trait drawn from a population, so some
participants follow their preference almost every trial and others are close to
indifferent. The evidence is the simplest current one (distance of the
alternation rate from a personal ideal); the change is in the decision rule.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        seq = seq.strip().upper()
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    return {"alt_rate_a": alternation_rate(sequence_a), "alt_rate_b": alternation_rate(sequence_b)}


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal alternation rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    # Sensitivity; a constrained prior keeps it from trading off against lapses.
    beta = pm.LogNormal("beta", mu=2.0, sigma=0.5)

    # Personal lapse (guessing) rate (non-centred logit-normal population).
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    score_a = -pt.sqr(alt_rate_a - theta)
    score_b = -pt.sqr(alt_rate_b - theta)
    p_engaged = pm.math.sigmoid(beta * (score_a - score_b))
    lam = lapse[participant_id]
    # Clipped so a saturated sigmoid with a near-zero lapse never gives log(0).
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
