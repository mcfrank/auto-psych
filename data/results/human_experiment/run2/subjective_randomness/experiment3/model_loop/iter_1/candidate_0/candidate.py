"""
Random-looking sequences are judged by an evidence accumulation process where each item provides a baseline weight of evidence, which is discounted by the sequence's deviation from a messy prototype. Rather than evaluating the global alternation rate, this prototype specifies an ideal run length, and each run incurs a penalty proportional to its squared deviation from this ideal length (alongside a penalty for overall imbalance), simultaneously punishing extreme alternation rates and disproportionately long local streaks.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return the sum of squared run lengths for each sequence."""
    def get_sum_sq_runs(seq):
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
        return float(sum(x**2 for x in runs))

    return {
        "sum_sq_runs_a": get_sum_sq_runs(sequence_a),
        "sum_sq_runs_b": get_sum_sq_runs(sequence_b)
    }

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    
    sum_sq_runs_a = pm.Data("sum_sq_runs_a", np.zeros(1, dtype="float64"))
    sum_sq_runs_b = pm.Data("sum_sq_runs_b", np.zeros(1, dtype="float64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    ideal_run = pm.Uniform("ideal_run", lower=1.0, upper=5.0)
    theta_imb = pm.Uniform("theta_imb", lower=0.0, upper=1.0)
    
    run_weight = pm.Uniform("run_weight", lower=0.01, upper=0.99)
    balance_weight = 1.0 - run_weight

    # Baseline evidence per item
    base_evidence = pm.HalfNormal("base_evidence", sigma=2.0)

    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # Casts
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")
    num_runs_a = pt.cast(alts_a, "float64") + 1.0
    num_runs_b = pt.cast(alts_b, "float64") + 1.0

    # The penalty for the run-length prototype is the sum of (L_i - ideal_run)^2 over all runs.
    # Expanded mathematically: sum(L_i^2) - 2 * ideal_run * sum(L_i) + num_runs * ideal_run^2
    # Since sum(L_i) is just the total length n
    run_penalty_a = sum_sq_runs_a - 2.0 * ideal_run * n_a_f + num_runs_a * (ideal_run ** 2)
    run_penalty_b = sum_sq_runs_b - 2.0 * ideal_run * n_b_f + num_runs_b * (ideal_run ** 2)

    # Because this penalty scales with sequence length (it's a sum over runs),
    # we average it per-item to combine with the item-level baseline evidence,
    # similar to how inner_loop_model computes average rep/alt penalties.
    avg_run_penalty_a = run_penalty_a / pt.clip(n_a_f, 1.0, np.inf)
    avg_run_penalty_b = run_penalty_b / pt.clip(n_b_f, 1.0, np.inf)

    # Quadratic distance for messy imbalance prototype
    imb_penalty_a = (imbalance_a - theta_imb) ** 2
    imb_penalty_b = (imbalance_b - theta_imb) ** 2

    # Evidence accumulation: length * (baseline - penalty)
    score_a = n_a_f * (
        base_evidence - (balance_weight * imb_penalty_a + run_weight * avg_run_penalty_a)
    )
    score_b = n_b_f * (
        base_evidence - (balance_weight * imb_penalty_b + run_weight * avg_run_penalty_b)
    )

    # Sigmoid link to probability, clamped for numerical safety
    p_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_raw, 1e-6, 1.0 - 1e-6))

    # Likelihood
    pm.Bernoulli("response", p=p_left, observed=chose_left)
