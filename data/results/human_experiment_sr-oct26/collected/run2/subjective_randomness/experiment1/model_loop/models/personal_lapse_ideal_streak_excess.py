"""Personal alternation ideal with a personal lapse rate and an absolute-streak penalty.

Refinement of `personal_ideal_with_personal_lapse`: each person judges a
sequence as random by how close its proportion of H/T switches is to their own
ideal switching rate, and lapses to a random pick at their own rate. The single
change: when engaged, people also penalise the longest streak by the number of
flips it exceeds two (HHH costs 1, HHHH costs 2, ...), regardless of length.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    def streak_excess(seq):
        best = cur = 1
        for x, y in zip(seq, seq[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return float(max(best - 2, 0))

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "streak_excess_a": streak_excess(a),
        "streak_excess_b": streak_excess(b),
    }


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    str_a = pm.Data("streak_excess_a", np.zeros(1, dtype="float64"))
    str_b = pm.Data("streak_excess_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal switching rate (population on the logit scale, non-centred).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to squared distance from the personal ideal.
    beta = pm.HalfNormal("beta", sigma=10.0)
    # Shared penalty per flip of the longest streak beyond two.
    delta = pm.Normal("delta", mu=0.0, sigma=1.0)

    # Personal lapse rate (population on the logit scale, non-centred).
    mu_lapse = pm.Normal("mu_lapse", mu=-2.5, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=N_SLOTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    eps = lapse[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    p_engaged = pm.math.sigmoid(beta * (dist_b - dist_a) + delta * (str_b - str_a))
    p_left = pm.Deterministic("p_left", 0.5 * eps + (1.0 - eps) * p_engaged)

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
