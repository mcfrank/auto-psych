"""
People judge the randomness of a sequence by comparing its maximum run proportion to a subjective ideal proportion, but following the Weber-Fechner law, their perception of this difference is logarithmic, penalizing the absolute log-ratio of the observed run proportion to the ideal proportion.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    
    # Cognitive parameters
    # The ideal maximum run proportion (bounded between 0 and 1)
    # A Beta(2, 5) prior suggests an expected ideal proportion around 2/7 (~0.28)
    ideal_prop = pm.Beta("ideal_prop", alpha=2.0, beta=5.0)
    
    # Sensitivity to the log-ratio deviation
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # Safe log calculations to avoid -inf if values drop to exactly 0
    safe_norm_a = pt.clip(max_run_norm_a, 1e-4, 1.0)
    safe_norm_b = pt.clip(max_run_norm_b, 1e-4, 1.0)
    safe_ideal = pt.clip(ideal_prop, 1e-4, 1.0)
    
    # Calculate the penalty for each sequence as the absolute log-ratio
    # |log(obs) - log(ideal)| = |log(obs / ideal)|
    penalty_a = pt.abs(pt.log(safe_norm_a) - pt.log(safe_ideal))
    penalty_b = pt.abs(pt.log(safe_norm_b) - pt.log(safe_ideal))
    
    # Probability of choosing left (sequence A)
    # The sequence with the smaller penalty is perceived as more random.
    # Therefore, p_left should be large when penalty_a < penalty_b.
    p_left_raw = pm.math.sigmoid(tau * (penalty_b - penalty_a))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
