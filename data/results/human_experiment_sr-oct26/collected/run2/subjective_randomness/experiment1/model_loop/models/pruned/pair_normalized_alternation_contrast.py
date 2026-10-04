"""Pair-normalised alternation contrast.

People judge the pair relative to itself: each person measures how far each
sequence's switching rate is from their own ideal switching rate, and the
difference between the two distances is divided by how far both sequences are
from that ideal together (divisive contrast normalisation). A given gap is
decisive when both sequences are near the ideal and weak when both are far,
so the same sequence is judged differently beside a different partner.
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

    # Population distribution of personal ideal switching rates (logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to the normalised contrast, and the semi-saturation constant
    # of the divisive normalisation (small: strongly pair-relative).
    beta = pm.HalfNormal("beta", sigma=5.0)
    log_c = pm.Normal("log_c", mu=np.log(0.1), sigma=1.0)
    c = pm.Deterministic("c", pm.math.exp(log_c))

    theta = ideal[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    contrast = (dist_b - dist_a) / (c + dist_a + dist_b)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * contrast))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
