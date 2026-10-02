"""Edge-streak salience added to the lapse / individual-balance / ideal-alternation model.

Refinement of `lapse_individual_balance_alternation`: each person judges a
sequence as more random the closer its alternation rate lies to their own ideal
switching rate, penalises heads/tails imbalance by a weight of their own, and
guesses on some trials at a personal lapse rate. The one change: a run of
identical flips at the start or end of a sequence is salient, and the number of
flips by which the longer edge run exceeds two makes the sequence look less
random by a shared weight (a streak in the middle carries no extra penalty).
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

    def edge_streak(seq):
        edge = max(run_from_start(seq), run_from_start(seq[::-1]))
        return float(max(edge - 2, 0))

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "imbalance_a": imbalance(a),
        "imbalance_b": imbalance(b),
        "edge_streak_a": edge_streak(a),
        "edge_streak_b": edge_streak(b),
    }


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    edge_streak_a = pm.Data("edge_streak_a", np.zeros(1, dtype="float64"))
    edge_streak_b = pm.Data("edge_streak_b", np.zeros(1, dtype="float64"))
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

    # Shared penalty per flip of edge streak beyond two.
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.0)

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    g = gamma[participant_id]
    score_a = -beta * pt.sqr(alt_rate_a - theta) - g * imbalance_a - kappa * edge_streak_a
    score_b = -beta * pt.sqr(alt_rate_b - theta) - g * imbalance_b - kappa * edge_streak_b
    p_engaged = pm.math.sigmoid(score_a - score_b)
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
