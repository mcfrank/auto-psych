# file: alternation_heuristic.py
"""
A simple heuristic model where observers judge randomness based on the proportion 
of alternations (H↔T transitions). Sequences with a higher alternation rate are 
perceived as more random.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: alternation proportions for the two sequences
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    
    # Free cognitive parameter: softmax temperature on the difference in alternation rates
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    # The observer prefers the sequence with the higher alternation proportion
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (p_alts_a - p_alts_b)))
    
    # Observed responses: 1 = chose left (sequence A), 0 = chose right
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)