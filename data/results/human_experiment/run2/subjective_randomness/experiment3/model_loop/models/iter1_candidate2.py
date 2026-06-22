"""
Random-looking sequences are evaluated by their constituent streaks (runs), where each run incurs a cognitive penalty proportional to its squared deviation from an ideal 'messy' run length. Because this single penalty operates locally on every run, it naturally accounts for ideal alternation rates, strongly punishes disproportionately long clusters, and implicitly penalizes global imbalance without requiring separate tracking of these global features.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return new numeric feature columns for one stimulus pair."""

    def sum_sq_runs(seq):
        if not seq:
            return 0.0
        runs = []
        curr = seq[0]
        length = 1
        for c in seq[1:]:
            if c == curr:
                length += 1
            else:
                runs.append(length)
                curr = c
                length = 1
        runs.append(length)
        return float(sum(r**2 for r in runs))

    return {
        "sum_sq_runs_a": sum_sq_runs(sequence_a),
        "sum_sq_runs_b": sum_sq_runs(sequence_b),
    }


with pm.Model() as model:
    # Sequence lengths
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))

    # Alternations (number of runs is alts + 1)
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Sum of squared run lengths (from compute_features)
    sum_sq_runs_a = pm.Data("sum_sq_runs_a", np.zeros(1, dtype="float64"))
    sum_sq_runs_b = pm.Data("sum_sq_runs_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=1.0)
    ideal_len = pm.Normal("ideal_len", mu=1.5, sigma=1.0)

    k_a = alts_a + 1
    k_b = alts_b + 1

    # We compute the total penalty for a sequence: sum_{i=1}^{k} (L_i - ideal_len)^2
    # sum (L_i - ideal_len)^2 = sum(L_i^2) - 2 * ideal_len * sum(L_i) + k * ideal_len^2
    total_penalty_a = sum_sq_runs_a - 2.0 * ideal_len * n_a + k_a * (ideal_len**2)
    total_penalty_b = sum_sq_runs_b - 2.0 * ideal_len * n_b + k_b * (ideal_len**2)

    # Normalize by sequence length so the penalty is an average per-item cost
    norm_penalty_a = total_penalty_a / n_a
    norm_penalty_b = total_penalty_b / n_b

    # Sequences with lower penalty are perceived as more random
    # P(choose left) = sigmoid(tau * (penalty_b - penalty_a))
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (norm_penalty_b - norm_penalty_a))
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
