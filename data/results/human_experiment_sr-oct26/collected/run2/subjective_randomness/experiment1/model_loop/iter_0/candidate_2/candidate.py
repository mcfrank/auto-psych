"""Personal switch-rate ideal.

Each person carries their own internal ideal of how often a random coin should
switch between heads and tails, and judges a sequence as more random the closer
its switch rate comes to that personal ideal. People differ in where this ideal
sits, so the same pair can be judged in opposite directions by different people.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def switch_rate(seq):
        seq = seq.strip().upper()
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    return {"switch_a": switch_rate(sequence_a), "switch_b": switch_rate(sequence_b)}


N_SLOTS = 400

with pm.Model() as model:
    switch_a = pm.Data("switch_a", np.zeros(1, dtype="float64"))
    switch_b = pm.Data("switch_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of personal ideal switch rates (logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic(
        "ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal)
    )

    # Sensitivity to squared distance from the ideal.
    beta = pm.HalfNormal("beta", sigma=10.0)

    rho = ideal[participant_id]
    score_diff = (switch_b - rho) ** 2 - (switch_a - rho) ** 2
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * score_diff))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
