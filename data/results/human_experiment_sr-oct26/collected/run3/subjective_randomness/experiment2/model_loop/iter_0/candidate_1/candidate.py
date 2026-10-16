"""People judge randomness by one gist cue, the sequence's switch rate (share of
adjacent flips that differ), but each person has their own signed preference for
switching drawn from a population that may straddle zero: most see frequent
switching as random, some see streaky sequences as random."""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def switch_rate(seq):
        seq = seq.strip().upper()
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    return {"switch_diff": switch_rate(sequence_a) - switch_rate(sequence_b)}


with pm.Model() as model:
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    mu = pm.Normal("mu", mu=0.0, sigma=3.0)
    sigma = pm.HalfNormal("sigma", sigma=3.0)
    z = pm.Normal("z", mu=0.0, sigma=1.0, shape=400)
    beta = pm.Deterministic("beta", mu + sigma * z)

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta[participant_id] * switch_diff)
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
