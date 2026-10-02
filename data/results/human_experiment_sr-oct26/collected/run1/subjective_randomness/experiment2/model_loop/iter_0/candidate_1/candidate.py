"""Lempel-Ziv incompressibility with personal sensitivity.

People judge a coin-flip sequence's randomness by how hard it is to describe
compactly: a sequence looks random in proportion to how many genuinely new
chunks it takes to spell it out left to right (its LZ76 compression
complexity). How strongly incompressibility drives the choice is each person's
own trait, drawn from a population distribution.
"""

import numpy as np
import pymc as pm

MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def lz76_phrases(seq):
        # Kaspar-Schuster LZ76 exhaustive-history phrase count.
        n = len(seq)
        if n < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        i, c, k, l, k_max = 0, 1, 1, 1, 1
        while True:
            if seq[i + k - 1] == seq[l + k - 1]:
                k += 1
                if l + k > n:
                    c += 1
                    break
            else:
                k_max = max(k, k_max)
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    if l + 1 > n:
                        break
                    i, k, k_max = 0, 1, 1
                else:
                    k = 1
        return float(c)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {"lz_a": lz76_phrases(a), "lz_b": lz76_phrases(b)}


with pm.Model() as model:
    lz_a = pm.Data("lz_a", np.zeros(1, dtype="float64"))
    lz_b = pm.Data("lz_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal sensitivity to incompressibility (non-centred population).
    mu_beta = pm.Normal("mu_beta", mu=0.5, sigma=1.0)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    beta = mu_beta + sigma_beta * z_beta

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta[participant_id] * (lz_a - lz_b))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
