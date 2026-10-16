"""Most lopsided window alarm.

People judge randomness by scanning a sequence for its single most lopsided
stretch: among all contiguous windows of every length, they find the one whose
heads/tails imbalance would be most surprising for a fair coin, and that one
stretch alone sets how non-random the sequence looks. The sequence whose worst
stretch is less surprising is chosen as more random; people differ in how
decisively this drives their choice.
"""
import math

import numpy as np
import pymc as pm


def _two_sided_tail(n, k):
    """P(|X - n/2| >= |k - n/2|) for X ~ Binomial(n, 1/2)."""
    dev = abs(k - n / 2.0)
    total = sum(math.comb(n, j) for j in range(n + 1) if abs(j - n / 2.0) >= dev - 1e-9)
    return total / 2.0 ** n


def _max_window_surprise(seq):
    seq = seq.strip().upper()
    n = len(seq)
    best = 0.0
    for length in range(2, n + 1):
        for start in range(n - length + 1):
            k = seq[start:start + length].count("H")
            s = -math.log2(_two_sided_tail(length, k))
            if s > best:
                best = s
    return best


def compute_features(sequence_a, sequence_b):
    return {
        "max_surprise_a": _max_window_surprise(sequence_a),
        "max_surprise_b": _max_window_surprise(sequence_b),
    }


with pm.Model() as model:
    max_surprise_a = pm.Data("max_surprise_a", np.zeros(1, dtype="float64"))
    max_surprise_b = pm.Data("max_surprise_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific decisiveness (log-normal population, non-centred).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z = pm.Normal("z", mu=0.0, sigma=1.0, shape=400)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z)

    diff = max_surprise_b - max_surprise_a
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta[participant_id] * diff))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
