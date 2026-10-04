"""Personal alternation ideal with periodic and absolute-streak penalties.

Refinement of `periodic_penalized_alternation_ideal`: each person judges a
sequence as random by how close its proportion of H/T switches is to their own
ideal switching rate, plus a shared penalty for visibly periodic sequences.
The single change: a shared penalty on the absolute number of flips by which
the longest streak exceeds two (HHH costs 1, HHHH costs 2, ...), regardless of
sequence length.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    def periodic(seq):
        n = len(seq)
        for p in range(1, n // 2 + 1):
            if all(seq[i] == seq[i + p] for i in range(n - p)):
                return 1.0
        return 0.0

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
        "periodic_a": periodic(a),
        "periodic_b": periodic(b),
        "streak_excess_a": streak_excess(a),
        "streak_excess_b": streak_excess(b),
    }


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    per_a = pm.Data("periodic_a", np.zeros(1, dtype="float64"))
    per_b = pm.Data("periodic_b", np.zeros(1, dtype="float64"))
    str_a = pm.Data("streak_excess_a", np.zeros(1, dtype="float64"))
    str_b = pm.Data("streak_excess_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of personal ideal switching rates (logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to squared distance from the personal ideal.
    beta = pm.HalfNormal("beta", sigma=10.0)
    # Shared penalty for a visibly periodic sequence (positive: less random).
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)
    # Shared penalty per flip of the longest streak beyond two.
    delta = pm.Normal("delta", mu=0.0, sigma=1.0)

    theta = ideal[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    score = (
        beta * (dist_b - dist_a)
        + gamma * (per_b - per_a)
        + delta * (str_b - str_a)
    )
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
