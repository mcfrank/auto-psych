"""
People judge the randomness of a sequence according to the local representativeness heuristic, meaning they expect every short contiguous segment of the sequence to independently reflect the global properties of a fair coin (equal proportions of heads and tails, and a 50% alternation rate), and they perceive a sequence as less random based on its average deviation from these ideals across all local sliding windows.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Compute the average local deviation from 0.5 balance and 0.5 alternation rate for short sliding windows."""
    def local_rep_penalty(seq, w=6):
        seq = seq.strip().upper()
        if len(seq) < w:
            w = len(seq)
        if w < 2:
            return 0.0
            
        penalty = 0.0
        for i in range(len(seq) - w + 1):
            window = seq[i:i+w]
            h_prop = window.count('H') / w
            alts = sum(1 for j in range(1, w) if window[j] != window[j-1])
            alt_prop = alts / (w - 1)
            
            # Penalize absolute deviation from expected 0.5 balance and 0.5 alternation rate
            penalty += abs(h_prop - 0.5) + abs(alt_prop - 0.5)
            
        return penalty / (len(seq) - w + 1)

    return {
        "local_rep_penalty_a": local_rep_penalty(sequence_a),
        "local_rep_penalty_b": local_rep_penalty(sequence_b)
    }

with pm.Model() as model:
    # Inputs
    pen_a = pm.Data("local_rep_penalty_a", np.zeros(1, dtype="float64"))
    pen_b = pm.Data("local_rep_penalty_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter: how much local deviation decreases perceived randomness
    weight = pm.HalfNormal("weight", sigma=5.0)

    # Score is negative penalty (higher deviation from local representativeness = less random)
    score_a = -weight * pen_a
    score_b = -weight * pen_b

    # Numerically safe probability mapping
    p_left_raw = pm.math.sigmoid(score_a - score_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
