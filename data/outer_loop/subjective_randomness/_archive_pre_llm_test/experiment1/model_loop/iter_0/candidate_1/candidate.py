"""
People judge a sequence as less random the longer its longest continuous run of identical outcomes. When comparing two sequences, they prefer the one with the shorter maximum run length as being more random.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="float64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter for decision noise / sensitivity.
    tau = pm.HalfNormal("tau", sigma=1.0)

    # People prefer the sequence with the shorter max run, 
    # so longer runs decrease the subjective randomness score.
    score_a = -max_run_a
    score_b = -max_run_b
    
    # Sigmoid function maps differences in scores to probabilities.
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b))
    
    # Clip to avoid exact 0 or 1 for numerical stability during MCMC.
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
