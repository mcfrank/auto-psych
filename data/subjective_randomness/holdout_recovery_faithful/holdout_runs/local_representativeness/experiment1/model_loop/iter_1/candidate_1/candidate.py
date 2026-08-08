"""
People judge the randomness of a sequence by intuitively estimating its algorithmic compressibility; sequences that can be mentally parsed into fewer novel chunks—whether due to long identical streaks or rigidly repeating alternating patterns—are perceived as highly structured and therefore less random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return Lempel-Ziv complexity for both sequences."""
    def lz_complexity(s):
        s = s.strip().upper()
        if not s:
            return 0.0
        i = 0
        c = 1
        while i < len(s):
            k = 1
            while i + k <= len(s) and s[i:i+k] in s[:i+k-1]:
                k += 1
            i += k
            if i < len(s):
                c += 1
        return float(c)

    return {
        "lz_complexity_a": lz_complexity(sequence_a),
        "lz_complexity_b": lz_complexity(sequence_b)
    }

with pm.Model() as model:
    # Feature inputs from compute_features
    lz_a = pm.Data("lz_complexity_a", np.zeros(1, dtype="float64"))
    lz_b = pm.Data("lz_complexity_b", np.zeros(1, dtype="float64"))
    
    # Feature inputs from precomputed columns
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter: sensitivity to complexity difference
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Compute chunk rate: complexity normalized by sequence length
    rate_a = lz_a / pt.clip(pt.cast(n_a, "float64"), 1.0, np.inf)
    rate_b = lz_b / pt.clip(pt.cast(n_b, "float64"), 1.0, np.inf)

    # Higher rate means less compressible, so more random
    p_left_raw = pm.math.sigmoid(tau * (rate_a - rate_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
