"""
People evaluate the randomness of a sequence by intuitively parsing it into runs of identical outcomes and estimating the likelihood of these run lengths. Instead of a standard geometric expectation, they assume the length of any streak follows a subjective Poisson distribution, which inherently applies a severe factorial penalty to excessively long runs and seamlessly accounts for both streak aversion and the preference for alternations without mixing separate heuristics.
"""

import math
import numpy as np
import pymc as pm
import pytensor.tensor as pt

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return new numeric feature columns for one stimulus pair."""
    def get_s_fact(seq):
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
        
        return float(sum(math.log(math.factorial(r - 1)) for r in runs))
        
    return {
        "s_fact_a": get_s_fact(sequence_a),
        "s_fact_b": get_s_fact(sequence_b)
    }

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    s_fact_a = pm.Data("s_fact_a", np.zeros(1, dtype="float64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    s_fact_b = pm.Data("s_fact_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    lambda_run = pm.Gamma("lambda_run", alpha=2.0, beta=2.0)
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Derive variables for sequence A
    m_a = pt.cast(alts_a + 1, "float64")
    rep_a = pt.cast(n_a - 1 - alts_a, "float64")
    n_a_f = pt.cast(n_a, "float64")
    
    # Randomness score A = average log-likelihood per flip under the subjective Poisson-run model
    log_lambda = pt.log(lambda_run + 1e-6)
    ll_a = -m_a * lambda_run + rep_a * log_lambda - s_fact_a
    score_a = ll_a / pt.maximum(n_a_f, 1.0)

    # Derive variables for sequence B
    m_b = pt.cast(alts_b + 1, "float64")
    rep_b = pt.cast(n_b - 1 - alts_b, "float64")
    n_b_f = pt.cast(n_b, "float64")
    
    # Randomness score B
    ll_b = -m_b * lambda_run + rep_b * log_lambda - s_fact_b
    score_b = ll_b / pt.maximum(n_b_f, 1.0)

    # Choice probability
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
