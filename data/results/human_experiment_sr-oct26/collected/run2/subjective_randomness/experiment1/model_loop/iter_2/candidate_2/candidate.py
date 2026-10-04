"""Personal alternation ideal with a personal lapse rate.

People judge which sequence looks more random by how close its proportion of
H/T switches is to their own personal ideal switching rate, but the decision
rule is not always engaged: each person has their own lapse rate, the share of
trials on which they pick a side at random without evaluating the sequences.
The evidence is the simplest current model's (squared distance from a personal
ideal switch rate); the claim is the person-specific lapse in the decision rule.
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

    # Personal ideal switching rate (population on the logit scale, non-centred).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to squared distance from the personal ideal.
    beta = pm.HalfNormal("beta", sigma=10.0)

    # Personal lapse rate (population on the logit scale, non-centred).
    mu_lapse = pm.Normal("mu_lapse", mu=-2.5, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=N_SLOTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    eps = lapse[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    p_engaged = pm.math.sigmoid(beta * (dist_b - dist_a))
    p_left = pm.Deterministic("p_left", 0.5 * eps + (1.0 - eps) * p_engaged)

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
