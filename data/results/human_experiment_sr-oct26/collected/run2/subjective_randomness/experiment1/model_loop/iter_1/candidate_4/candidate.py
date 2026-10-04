"""Personal alternation ideal with a longest-streak penalty.

Refinement of `personal_alternation_ideal`: each person judges a sequence as
random to the extent that its proportion of H/T switches is close to their own
personal ideal switching rate, and additionally treats the longest streak of
identical outcomes (relative to sequence length) as a salient sign of
non-randomness. The one change is a shared penalty on the longest run.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    def longest_run_norm(seq):
        best = cur = 1
        for x, y in zip(seq, seq[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return (best - 1) / (len(seq) - 1)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "longest_run_a": longest_run_norm(a),
        "longest_run_b": longest_run_norm(b),
    }


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    run_a = pm.Data("longest_run_a", np.zeros(1, dtype="float64"))
    run_b = pm.Data("longest_run_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of personal ideal switching rates (logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to squared distance from the personal ideal.
    beta = pm.HalfNormal("beta", sigma=10.0)
    # Shared penalty on the longest streak (normalised by length).
    gamma = pm.Normal("gamma", mu=0.0, sigma=3.0)

    theta = ideal[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    score = beta * (dist_b - dist_a) + gamma * (run_b - run_a)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
