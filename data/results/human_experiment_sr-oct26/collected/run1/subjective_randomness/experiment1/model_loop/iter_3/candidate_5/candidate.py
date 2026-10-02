"""Personal lapse rate on a personal-ideal-alternation judgment with a streak penalty.

Refinement of `ideal_alternation_streak_penalty`: each person judges a sequence
as more random the closer its alternation rate lies to their own ideal switching
rate, and everyone penalises the length of a sequence's longest run beyond what
the alternation rate shows. The one change: on some trials the person guesses
instead of comparing, at a personal lapse rate drawn from a population.
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

    def longest_run(seq):
        seq = seq.strip().upper()
        best = run = 1
        for x, y in zip(seq, seq[1:]):
            run = run + 1 if x == y else 1
            best = max(best, run)
        return float(best)

    return {
        "alt_rate_a": alternation_rate(sequence_a),
        "alt_rate_b": alternation_rate(sequence_b),
        "max_run_a": longest_run(sequence_a),
        "max_run_b": longest_run(sequence_b),
    }


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="float64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal alternation rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    # Sensitivity to squared distance from the ideal.
    beta = pm.LogNormal("beta", mu=2.0, sigma=0.5)
    # Shared penalty per flip of the longest streak (positive: long runs look less random).
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)

    # Personal lapse (guessing) rate (non-centred logit-normal population).
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    score_a = -beta * pt.sqr(alt_rate_a - theta) - gamma * max_run_a
    score_b = -beta * pt.sqr(alt_rate_b - theta) - gamma * max_run_b
    p_engaged = pm.math.sigmoid(score_a - score_b)
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
