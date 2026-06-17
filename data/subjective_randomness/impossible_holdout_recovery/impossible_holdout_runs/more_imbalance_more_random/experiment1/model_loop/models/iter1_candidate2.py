"""
People judge sequence randomness based purely on the maximum run length heuristic. They assess the length of the longest continuous streak of identical outcomes in each sequence, perceiving the sequence with the shorter maximum streak as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter with a prior (inference fits it).
    # Sensitivity to differences in maximum run length.
    weight = pm.HalfNormal("weight", sigma=5.0)

    # Difference in max run length: if B has a longer run than A, A is more random (shorter run).
    # So max_run_b - max_run_a is positive, which should increase p_left.
    score_diff = weight * (pt.cast(max_run_b, "float64") - pt.cast(max_run_a, "float64"))
    
    # Clip probability to avoid numerical instability
    p_left_raw = pm.math.sigmoid(score_diff)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
