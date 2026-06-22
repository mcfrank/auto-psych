"""
People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but we refine how alternation typicality is evaluated. Rather than penalizing the sequence's global average alternation rate, people evaluate typicality at the level of individual runs (streaks): they accumulate a penalty for each run that grows quadratically with how much its length deviates from an ideal run length. This non-linear run-level penalty explains why sequences with atypically long streaks are judged as significantly less random even when their overall alternation rates match.
"""

import re
import numpy as np
import pymc as pm
import pytensor.tensor as pt

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return run-level features needed to compute the accumulated quadratic run penalty."""
    def get_runs(seq):
        seq = seq.strip().upper()
        if not seq: 
            return []
        return [len(m.group(0)) for m in re.finditer(r'(H+|T+)', seq)]
    
    def run_sq(runs):
        return sum(r**2 for r in runs)
        
    runs_a = get_runs(sequence_a)
    runs_b = get_runs(sequence_b)
    
    return {
        "run_sq_a": float(run_sq(runs_a)),
        "run_sq_b": float(run_sq(runs_b)),
        "n_runs_a": float(len(runs_a)),
        "n_runs_b": float(len(runs_b))
    }

with pm.Model() as model:
    # Stimulus inputs (precomputed and our custom run features)
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    run_sq_a = pm.Data("run_sq_a", np.zeros(1, dtype="float64"))
    n_runs_a = pm.Data("n_runs_a", np.zeros(1, dtype="float64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    run_sq_b = pm.Data("run_sq_b", np.zeros(1, dtype="float64"))
    n_runs_b = pm.Data("n_runs_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    ideal_p = pm.Beta("ideal_p", alpha=2.0, beta=2.0)
    ideal_run_len = pm.Uniform("ideal_run_len", lower=1.0, upper=5.0)

    # Weights for the deviations
    w_p = pm.HalfNormal("w_p", sigma=5.0)
    w_run = pm.HalfNormal("w_run", sigma=5.0)

    # Base typicality per event
    base_typ = pm.Normal("base_typ", mu=0.0, sigma=5.0)

    # Calculate empirical head proportion (safeguard against division by zero)
    p_a = h_a / pt.maximum(n_a, 1)
    p_b = h_b / pt.maximum(n_b, 1)

    # Calculate the run-level penalty: sum((L_i - ideal_run_len)^2)
    # Using the algebraic expansion: sum(L_i^2) - 2 * ideal_run_len * sum(L_i) + ideal_run_len^2 * n_runs
    # Note: sum(L_i) is exactly the total sequence length (n_a).
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")
    
    run_penalty_a = run_sq_a - 2.0 * ideal_run_len * n_a_f + pt.square(ideal_run_len) * n_runs_a
    run_penalty_b = run_sq_b - 2.0 * ideal_run_len * n_b_f + pt.square(ideal_run_len) * n_runs_b

    # Head proportion penalty is linear (absolute difference)
    p_penalty_a = pt.abs(p_a - ideal_p)
    p_penalty_b = pt.abs(p_b - ideal_p)

    # Total randomness score is the accumulated typicality over the sequence length,
    # minus the accumulated penalties for head proportion and run lengths.
    rand_a = n_a_f * base_typ - w_p * n_a_f * p_penalty_a - w_run * run_penalty_a
    rand_b = n_b_f * base_typ - w_p * n_b_f * p_penalty_b - w_run * run_penalty_b

    # Choice probability
    p_left_raw = pm.math.sigmoid(rand_a - rand_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
