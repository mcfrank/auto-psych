"""People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes relative to the total sequence length. However, their perception of this proportion is non-linear; they subjectively evaluate the normalized maximum run length according to a power law before comparing the sequences."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs for the normalized maximum run length
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    
    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=5.0)
    gamma = pm.HalfNormal("gamma", sigma=2.0)
    
    # Clip to avoid zero raised to a power less than 1, which can cause inf gradients
    safe_run_a = pt.clip(max_run_norm_a, 1e-6, 1.0)
    safe_run_b = pt.clip(max_run_norm_b, 1e-6, 1.0)
    
    # Non-linear power law perception of the streak proportion
    score_a = safe_run_a ** gamma
    score_b = safe_run_b ** gamma
    
    # Probability of choosing left (A)
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
