"""Switch-rate prototype judged with a sharp shared rule plus person-specific lapses.

People all judge randomness by how close a sequence's switch rate is to an
internal ideal switch rate, applying one shared, sharp decision rule; people
differ only in a lapse rate: on a person-specific fraction of trials they
ignore the judgment and pick a side at random (probability 1/2 each).
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def switch_rate(seq):
        seq = seq.strip().upper()
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    return {"sr_a": switch_rate(sequence_a), "sr_b": switch_rate(sequence_b)}


with pm.Model() as model:
    sr_a = pm.Data("sr_a", np.zeros(1, dtype="float64"))
    sr_b = pm.Data("sr_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Internal ideal switch rate.
    ideal = pm.Beta("ideal_rate", alpha=3.0, beta=2.0)
    # One shared decision sensitivity.
    log_beta = pm.Normal("log_beta", mu=2.0, sigma=1.0)
    beta = pm.math.exp(log_beta)

    # Person-specific lapse rate on the logit scale (non-centred population).
    mu_lapse = pm.Normal("mu_logit_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_logit_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=400)
    lapse = pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse)

    dist_a = (sr_a - ideal) ** 2
    dist_b = (sr_b - ideal) ** 2
    p_judge = pm.math.sigmoid(beta * (dist_b - dist_a))
    lam = lapse[participant_id]
    p_left = pm.Deterministic("p_left", 0.5 * lam + (1.0 - lam) * p_judge)

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
