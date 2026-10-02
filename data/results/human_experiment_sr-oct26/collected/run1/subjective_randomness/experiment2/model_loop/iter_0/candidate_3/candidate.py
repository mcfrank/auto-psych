"""Incumbent individual_streak_aversion_lapse plus a separately weighted final streak.

Refinement of `individual_streak_aversion_lapse`: each person judges a sequence
as more random the closer its alternation rate lies to their own ideal switching
rate, penalises heads/tails imbalance and the longest run by weights of their
own and guesses on some trials at a personal lapse rate. The one change: the run
of identical flips the sequence ends on (the last thing read) shifts its
apparent randomness by a shared weight of its own, lenient or harsh.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    def imbalance(seq):
        return abs(seq.count("H") - seq.count("T")) / len(seq)

    def longest_run_share(seq):
        best = cur = 1
        for x, y in zip(seq, seq[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return (best - 1) / (len(seq) - 1)

    def final_run_share(seq):
        cur = 1
        for x, y in zip(seq[-2::-1], seq[:0:-1]):
            if x != y:
                break
            cur += 1
        return (cur - 1) / (len(seq) - 1)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "imbalance_a": imbalance(a),
        "imbalance_b": imbalance(b),
        "maxrun_a": longest_run_share(a),
        "maxrun_b": longest_run_share(b),
        "endrun_a": final_run_share(a),
        "endrun_b": final_run_share(b),
    }


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    maxrun_a = pm.Data("maxrun_a", np.zeros(1, dtype="float64"))
    maxrun_b = pm.Data("maxrun_b", np.zeros(1, dtype="float64"))
    endrun_a = pm.Data("endrun_a", np.zeros(1, dtype="float64"))
    endrun_b = pm.Data("endrun_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal alternation rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    beta = pm.LogNormal("beta", mu=2.5, sigma=0.5)

    # Personal weight on H/T imbalance.
    mu_gamma = pm.Normal("mu_gamma", mu=0.0, sigma=2.0)
    sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=1.5)
    z_gamma = pm.Normal("z_gamma", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    gamma = mu_gamma + sigma_gamma * z_gamma

    # Personal streak aversion (weight on the longest run's share of the sequence).
    mu_delta = pm.Normal("mu_delta", mu=0.0, sigma=2.0)
    sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.5)
    z_delta = pm.Normal("z_delta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    delta = mu_delta + sigma_delta * z_delta

    # Shared extra weight on the final run (positive: end streaks penalised more).
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    g = gamma[participant_id]
    d = delta[participant_id]
    score_a = -beta * pt.sqr(alt_rate_a - theta) - g * imbalance_a - d * maxrun_a - kappa * endrun_a
    score_b = -beta * pt.sqr(alt_rate_b - theta) - g * imbalance_b - d * maxrun_b - kappa * endrun_b
    p_engaged = pm.math.sigmoid(score_a - score_b)
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
