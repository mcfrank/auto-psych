"""Personal alternation ideal.

Each person carries their own ideal switching rate for a random coin and
judges a sequence as random to the extent that its proportion of H/T switches
is close to that personal ideal. People differ in that ideal (some expect a
fair-coin rate near one half, others expect heavy over-alternation), which is
where this model departs from a single shared alternation prototype: on pairs
that both alternate heavily, strong over-alternators pick the more alternating
sequence while moderate people pick the other.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        seq = seq.strip().upper()
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    return {"alt_rate_a": alternation_rate(sequence_a), "alt_rate_b": alternation_rate(sequence_b)}


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of personal ideal switching rates (logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to squared distance from the personal ideal.
    beta = pm.HalfNormal("beta", sigma=10.0)

    theta = ideal[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (dist_b - dist_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
