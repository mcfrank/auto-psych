"""
People judge the randomness of a sequence by the diversity of its run lengths, perceiving sequences as more random when they contain an unpredictable mix of short and long streaks, which is evaluated as the Shannon entropy of the sequence's run-length distribution.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from collections import Counter
import math

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Compute the Shannon entropy of the run-length distribution."""
    def rle_entropy(seq):
        seq = seq.strip().upper()
        if not seq:
            return 0.0
        
        runs = []
        current = seq[0]
        count = 1
        for char in seq[1:]:
            if char == current:
                count += 1
            else:
                runs.append(count)
                current = char
                count = 1
        runs.append(count)
        
        c = Counter(runs)
        total = sum(c.values())
        ent = 0.0
        for cnt in c.values():
            p = cnt / total
            ent -= p * math.log2(p)
        return float(ent)

    return {
        "run_entropy_a": rle_entropy(sequence_a),
        "run_entropy_b": rle_entropy(sequence_b)
    }

with pm.Model() as model:
    # Stimulus inputs
    run_entropy_a = pm.Data("run_entropy_a", np.zeros(1, dtype="float64"))
    run_entropy_b = pm.Data("run_entropy_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)

    # Score is proportional to the run-length entropy
    score_a = beta * run_entropy_a
    score_b = beta * run_entropy_b

    # Choice probability
    p_left_raw = pm.math.sigmoid(score_a - score_b + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
