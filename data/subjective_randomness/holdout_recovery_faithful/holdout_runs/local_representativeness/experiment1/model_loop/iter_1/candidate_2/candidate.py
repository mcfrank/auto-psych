"""
People judge randomness according to the local representativeness heuristic by accumulating evaluations across the sequence; specifically, they assess every short sliding window for its deviation from ideal balance and ideal alternation rate, and sum these local representative scores, meaning longer sequences can be judged more random overall but are consistently penalized for unrepresentative local patches.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    def extract_local(seq, w=5):
        seq = seq.strip().upper()
        n = len(seq)
        if n < w:
            w = n if n > 0 else 1
            
        windows = n - w + 1 if n >= w else 0
        if windows == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        sum_imb_sq = 0.0
        sum_alt = 0.0
        sum_alt_sq = 0.0
        
        for i in range(windows):
            window = seq[i:i+w]
            heads = window.count('H')
            alts = sum(1 for j in range(w-1) if window[j] != window[j+1])
            
            imb = abs(heads / w - 0.5)
            sum_imb_sq += imb ** 2
            
            p_alt = alts / (w - 1) if w > 1 else 0.0
            sum_alt += p_alt
            sum_alt_sq += p_alt ** 2
            
        return float(windows), sum_imb_sq, sum_alt, sum_alt_sq

    win_a, imb_sq_a, alt_a, alt_sq_a = extract_local(sequence_a)
    win_b, imb_sq_b, alt_b, alt_sq_b = extract_local(sequence_b)
    
    return {
        "n_win_a": win_a,
        "sum_local_imb_sq_a": imb_sq_a,
        "sum_local_alt_a": alt_a,
        "sum_local_alt_sq_a": alt_sq_a,
        "n_win_b": win_b,
        "sum_local_imb_sq_b": imb_sq_b,
        "sum_local_alt_b": alt_b,
        "sum_local_alt_sq_b": alt_sq_b,
    }

with pm.Model() as model:
    # Inputs
    n_win_a = pm.Data("n_win_a", np.zeros(1, dtype="float64"))
    sum_local_imb_sq_a = pm.Data("sum_local_imb_sq_a", np.zeros(1, dtype="float64"))
    sum_local_alt_a = pm.Data("sum_local_alt_a", np.zeros(1, dtype="float64"))
    sum_local_alt_sq_a = pm.Data("sum_local_alt_sq_a", np.zeros(1, dtype="float64"))

    n_win_b = pm.Data("n_win_b", np.zeros(1, dtype="float64"))
    sum_local_imb_sq_b = pm.Data("sum_local_imb_sq_b", np.zeros(1, dtype="float64"))
    sum_local_alt_b = pm.Data("sum_local_alt_b", np.zeros(1, dtype="float64"))
    sum_local_alt_sq_b = pm.Data("sum_local_alt_sq_b", np.zeros(1, dtype="float64"))

    # Free parameters
    w_base = pm.HalfNormal("w_base", sigma=1.0)
    w_imb = pm.HalfNormal("w_imb", sigma=5.0)
    w_alt = pm.HalfNormal("w_alt", sigma=5.0)
    ideal_alt = pm.Beta("ideal_alt", alpha=6.0, beta=4.0)

    # Compute sequence scores
    sum_alt_dev_sq_a = sum_local_alt_sq_a - 2.0 * ideal_alt * sum_local_alt_a + n_win_a * (ideal_alt ** 2)
    score_a = n_win_a * w_base - w_imb * sum_local_imb_sq_a - w_alt * sum_alt_dev_sq_a

    sum_alt_dev_sq_b = sum_local_alt_sq_b - 2.0 * ideal_alt * sum_local_alt_b + n_win_b * (ideal_alt ** 2)
    score_b = n_win_b * w_base - w_imb * sum_local_imb_sq_b - w_alt * sum_alt_dev_sq_b

    # Choice probability
    diff = score_a - score_b
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(diff))
    p_left_safe = pt.clip(p_left, 1e-6, 1.0 - 1e-6)

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
