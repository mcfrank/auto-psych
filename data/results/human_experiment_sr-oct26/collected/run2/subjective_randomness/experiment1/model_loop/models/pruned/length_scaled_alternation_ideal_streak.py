"""Length-scaled personal alternation ideal with a longest-streak penalty.

Refinement of `personal_alternation_ideal_streak_penalty`: each person judges a
sequence as random to the extent that its proportion of H/T switches is close
to their own ideal switching rate, and a long streak counts against it. The one
change: the alternation impression accumulates over the transitions seen, so
sensitivity to the distance from the personal ideal grows as a fitted power of
the number of transitions (longer sequences give more weight to the same
rate departure).
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
        # Transitions relative to the longest sequences (8 flips, 7 transitions).
        "log_transitions": float(np.log((len(a) - 1) / 7.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    run_a = pm.Data("longest_run_a", np.zeros(1, dtype="float64"))
    run_b = pm.Data("longest_run_b", np.zeros(1, dtype="float64"))
    log_trans = pm.Data("log_transitions", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of personal ideal switching rates (logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to squared distance from the personal ideal at 7 transitions,
    # scaled by (transitions / 7) ** kappa: evidence accumulating with length.
    beta = pm.HalfNormal("beta", sigma=10.0)
    kappa = pm.Normal("kappa", mu=0.5, sigma=1.0)
    # Shared penalty on the longest streak (normalised by length).
    gamma = pm.Normal("gamma", mu=0.0, sigma=3.0)

    theta = ideal[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    sens = beta * pm.math.exp(kappa * log_trans)
    score = sens * (dist_b - dist_a) + gamma * (run_b - run_a)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
