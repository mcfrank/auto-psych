"""
People evaluate the randomness of a sequence solely by assessing whether its longest streak of identical outcomes is typical for a random sequence of that length. They linearly penalize sequences based on the absolute deviation of their observed maximum run length from the expected maximum run length of a fair coin (approximately log2 of the sequence length).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    n_a = pm.Data("n_a", np.zeros(1, dtype="float64"))
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="float64"))
    
    n_b = pm.Data("n_b", np.zeros(1, dtype="float64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter with a prior
    tau = pm.HalfNormal("tau", sigma=1.0)
    
    # Clip n to avoid negative or zero values in logarithm
    safe_n_a = pt.clip(n_a, 2.0, np.inf)
    safe_n_b = pt.clip(n_b, 2.0, np.inf)
    
    # Expected maximum run length approximation (log2(n))
    expected_max_run_a = pt.log(safe_n_a) / np.log(2.0)
    expected_max_run_b = pt.log(safe_n_b) / np.log(2.0)
    
    # Linearly penalize absolute deviation from expected max run
    score_a = -tau * pt.abs(max_run_a - expected_max_run_a)
    score_b = -tau * pt.abs(max_run_b - expected_max_run_b)
    
    # Probability of choosing sequence A (left) over sequence B (right)
    # Clip to avoid nan in Bernoulli likelihood
    p_left = pm.Deterministic("p_left", pt.clip(pm.math.sigmoid(score_a - score_b), 1e-6, 1 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
