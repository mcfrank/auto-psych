"""
People judge randomness by searching for local "clumps" of identical outcomes, specifically associating moderate-length runs (pairs and triplets) with the natural clumpiness of a stochastic process. Sequences are perceived as more random when a higher proportion of their outcomes belong to these moderate-length clusters, as this simultaneously avoids the artificial regularity of strict alternation and the perceived non-randomness of overly long streaks.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return new numeric feature columns for one stimulus pair."""
    def clumpiness(seq):
        seq = seq.strip().upper()
        if not seq:
            return 0.0
        runs = []
        current_run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        # Proportion of items that belong to runs of length 2 or 3
        clump_items = sum(r for r in runs if r in (2, 3))
        return float(clump_items / len(seq))
        
    return {
        "clumpiness_a": clumpiness(sequence_a),
        "clumpiness_b": clumpiness(sequence_b)
    }

with pm.Model() as model:
    # Stimulus inputs
    clumpiness_a = pm.Data("clumpiness_a", np.zeros(1, dtype="float64"))
    clumpiness_b = pm.Data("clumpiness_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Score is directly the clumpiness feature
    score_a = clumpiness_a
    score_b = clumpiness_b

    # Probability of choosing sequence A (left)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
