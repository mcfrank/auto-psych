"""Separate primacy and recency streak penalties on the lapse / balance / ideal-alternation model.

Refinement of `edge_streak_balance_alternation_lapse`: each person judges a
sequence as more random the closer its alternation rate lies to their own ideal
switching rate, penalises heads/tails imbalance by a weight of their own, and
guesses on some trials at a personal lapse rate; streaks at the edges carry an
extra penalty. The one change: the opening run and the closing run (the number
of flips by which each exceeds two) are penalised by separate shared weights,
so a streak read last (recency) can count for more than one read first.
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

    def run_from_start(seq):
        n = 1
        while n < len(seq) and seq[n] == seq[0]:
            n += 1
        return n

    def start_streak(seq):
        return float(max(run_from_start(seq) - 2, 0))

    def end_streak(seq):
        return float(max(run_from_start(seq[::-1]) - 2, 0))

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "imbalance_a": imbalance(a),
        "imbalance_b": imbalance(b),
        "start_streak_a": start_streak(a),
        "start_streak_b": start_streak(b),
        "end_streak_a": end_streak(a),
        "end_streak_b": end_streak(b),
    }


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    start_streak_a = pm.Data("start_streak_a", np.zeros(1, dtype="float64"))
    start_streak_b = pm.Data("start_streak_b", np.zeros(1, dtype="float64"))
    end_streak_a = pm.Data("end_streak_a", np.zeros(1, dtype="float64"))
    end_streak_b = pm.Data("end_streak_b", np.zeros(1, dtype="float64"))
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

    # Shared penalties per flip of streak beyond two: opening run (primacy)
    # and closing run (recency), each with its own weight.
    kappa_start = pm.Normal("kappa_start", mu=0.0, sigma=1.0)
    kappa_end = pm.Normal("kappa_end", mu=0.0, sigma=1.0)

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    g = gamma[participant_id]
    score_a = (
        -beta * pt.sqr(alt_rate_a - theta)
        - g * imbalance_a
        - kappa_start * start_streak_a
        - kappa_end * end_streak_a
    )
    score_b = (
        -beta * pt.sqr(alt_rate_b - theta)
        - g * imbalance_b
        - kappa_start * start_streak_b
        - kappa_end * end_streak_b
    )
    p_engaged = pm.math.sigmoid(score_a - score_b)
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
