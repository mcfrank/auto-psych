"""Length-scaled personal alternation ideal with a periodic-pattern penalty.

Refinement of `periodic_penalized_alternation_ideal`: each person judges a
sequence as random by how close its proportion of H/T switches is to their own
ideal switching rate, plus a shared penalty for visibly periodic sequences.
The single change: the evidence from switch rate accumulates with the number of
transitions seen, so the sensitivity to distance from the personal ideal scales
as a fitted power of the number of transitions (n - 1).
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

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "periodic_a": periodic(a),
        "periodic_b": periodic(b),
        # log of number of transitions relative to 4 (a mid-range length).
        "log_rel_transitions": float(np.log((len(a) - 1) / 4.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    per_a = pm.Data("periodic_a", np.zeros(1, dtype="float64"))
    per_b = pm.Data("periodic_b", np.zeros(1, dtype="float64"))
    log_rel_n = pm.Data("log_rel_transitions", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    beta = pm.HalfNormal("beta", sigma=10.0)
    # Exponent of transition count in the sensitivity (0: incumbent).
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.0)
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)

    theta = ideal[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    sens = beta * pm.math.exp(kappa * log_rel_n)
    score = sens * (dist_b - dist_a) + gamma * (per_b - per_a)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
