"""Personal response-side bias on a simple alternation-ideal judgement.

Each person has their own habitual lean toward the left or the right button,
which adds to their impression of which sequence is more random on every trial
(direction and strength differ across people, unrelated to the sequences). The
impression is simple: a sequence looks random to the extent that its proportion
of H/T switches is close to the person's own ideal switching rate.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {"alt_rate_a": alternation_rate(a), "alt_rate_b": alternation_rate(b)}


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal switching rate (non-centred, logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Shared sensitivity to distance from the ideal.
    beta = pm.HalfNormal("beta", sigma=10.0)

    # Personal response-side bias (positive: leans toward Left).
    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.5)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.5)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = mu_side + sigma_side * z_side

    theta = ideal[participant_id]
    evidence = beta * ((alt_b - theta) ** 2 - (alt_a - theta) ** 2)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(evidence + side[participant_id]))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
