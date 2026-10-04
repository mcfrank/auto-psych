"""Chunk-entropy variety judgement of randomness.

People judge randomness by the variety of short patterns a sequence contains:
read as overlapping chunks of one to four flips, a sequence looks random to the
extent that its chunks of each size are spread evenly over the possible chunks
(normalised chunk entropy, averaged over chunk sizes), and repetitive to the
extent that the same few chunks recur — a lopsided H/T count, a long streak,
strict alternation, or an almost-repeating unit. People differ in how strongly
this felt variety drives their choice.
"""
import math
from collections import Counter

import numpy as np
import pymc as pm

MAX_CHUNK = 4


def compute_features(sequence_a, sequence_b):
    def variety(seq):
        n = len(seq)
        scores = []
        for k in range(1, min(MAX_CHUNK, n) + 1):
            chunks = [seq[i:i + k] for i in range(n - k + 1)]
            possible = min(2 ** k, len(chunks))
            if possible < 2:
                continue
            counts = np.array(list(Counter(chunks).values()), dtype=float)
            p = counts / counts.sum()
            entropy = float(-(p * np.log2(p)).sum())
            scores.append(entropy / math.log2(possible))
        return float(np.mean(scores)) if scores else 0.0

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {"chunk_entropy_a": variety(a), "chunk_entropy_b": variety(b)}


N_SLOTS = 400

with pm.Model() as model:
    ent_a = pm.Data("chunk_entropy_a", np.zeros(1, dtype="float64"))
    ent_b = pm.Data("chunk_entropy_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific sensitivity to chunk variety (log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=1.2, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.Deterministic("beta", pm.math.exp(mu_log_beta + sigma_log_beta * z_beta))

    score = beta[participant_id] * (ent_a - ent_b)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
