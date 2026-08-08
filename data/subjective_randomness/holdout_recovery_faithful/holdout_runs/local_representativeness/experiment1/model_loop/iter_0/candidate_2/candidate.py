"""
People judge the randomness of a sequence by comparing the Shannon entropy of its run-length distribution to a subjective ideal, penalizing sequences whose streaks are either too rigidly structured (like the zero-entropy `HHTTHHTT`) or misaligned with their mental model of a fair coin. This disagrees sharply with the incumbent evidence-accumulation model by heavily penalizing artificially periodic sequences that have a typical total count of runs but lack realistic run-length diversity.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Compute the Shannon entropy of the run-length distribution for each sequence."""
    from collections import Counter
    import math

    def run_length_entropy(seq: str) -> float:
        seq = seq.strip().upper()
        if not seq:
            return 0.0
        
        # Parse run lengths
        runs = []
        current_char = seq[0]
        current_len = 1
        for c in seq[1:]:
            if c == current_char:
                current_len += 1
            else:
                runs.append(current_len)
                current_char = c
                current_len = 1
        runs.append(current_len)
        
        # Compute Shannon entropy of the run lengths
        counts = Counter(runs)
        total_runs = len(runs)
        entropy = 0.0
        for count in counts.values():
            p = count / total_runs
            entropy -= p * math.log(p)
            
        return float(entropy)

    return {
        "run_entropy_a": run_length_entropy(sequence_a),
        "run_entropy_b": run_length_entropy(sequence_b)
    }

with pm.Model() as model:
    # Read the custom features
    run_entropy_a = pm.Data("run_entropy_a", np.zeros(1, dtype="float64"))
    run_entropy_b = pm.Data("run_entropy_b", np.zeros(1, dtype="float64"))
    
    # Free cognitive parameters
    ideal_entropy = pm.HalfNormal("ideal_entropy", sigma=2.0)
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    # Penalty is the squared deviation from the ideal run-length entropy
    penalty_a = (run_entropy_a - ideal_entropy) ** 2
    penalty_b = (run_entropy_b - ideal_entropy) ** 2
    
    # Calculate response probability (lower penalty = more random = more likely to choose)
    score_diff = tau * (penalty_b - penalty_a)
    score_diff_safe = pt.clip(score_diff, -20.0, 20.0)
    
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_diff_safe))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
