"""People judge the randomness of a sequence by assessing how much the absolute length of its longest streak of identical outcomes exceeds the natural logarithmic growth expected for a sequence of that total length."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    
    # Free cognitive parameter
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # Cast to float
    run_a = pt.cast(max_run_a, "float64")
    run_b = pt.cast(max_run_b, "float64")
    len_a = pt.cast(n_a, "float64")
    len_b = pt.cast(n_b, "float64")
    
    # Expected run length grows as log2(N)
    log2 = np.log(2.0)
    expected_run_a = pt.log(pt.maximum(len_a, 1.0)) / log2
    expected_run_b = pt.log(pt.maximum(len_b, 1.0)) / log2
    
    # Score is the difference between actual maximum run and expected maximum run
    score_a = run_a - expected_run_a
    score_b = run_b - expected_run_b
    
    # Probability of choosing left (A)
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
