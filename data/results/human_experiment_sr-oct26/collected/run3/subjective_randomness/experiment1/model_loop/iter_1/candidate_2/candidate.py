"""Ideal switch-rate prototype.

People judge randomness by a single gist cue: how often the sequence switches
between heads and tails. They hold an internal ideal switch rate, and a
sequence looks random to the extent its switch rate is close to that ideal --
too few switches (long streaks) and too many (perfect alternation) both make
it look less random. People differ in how decisively this cue drives choice.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def switch_rate(seq):
        seq = seq.strip().upper()
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    return {"switch_rate_a": switch_rate(sequence_a), "switch_rate_b": switch_rate(sequence_b)}


with pm.Model() as model:
    rate_a = pm.Data("switch_rate_a", np.zeros(1, dtype="float64"))
    rate_b = pm.Data("switch_rate_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Internal ideal switch rate.
    ideal = pm.Beta("ideal_rate", alpha=3.0, beta=2.0)

    # Person-specific decisiveness, log-normal population (non-centred).
    mu_log_beta = pm.Normal("mu_log_beta", mu=2.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pm.Deterministic("beta", pm.math.exp(mu_log_beta + sigma_log_beta * z))

    dist_a = (rate_a - ideal) ** 2
    dist_b = (rate_b - ideal) ** 2
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta[participant_id] * (dist_b - dist_a))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
