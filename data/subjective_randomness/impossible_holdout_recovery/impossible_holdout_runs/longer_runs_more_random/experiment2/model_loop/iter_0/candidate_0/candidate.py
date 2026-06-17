"""People judge the randomness of a sequence by focusing solely on its longest streak of identical outcomes. They evaluate this streak as a simple ratio of the maximum run length to the total sequence length, perceiving a sequence as more random the larger this absolute fraction is."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    
    # Free cognitive parameter for sensitivity
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # Perceived randomness is the simple ratio of maximum run to total length
    score_a = pt.cast(max_run_a, "float64") / pt.cast(n_a, "float64")
    score_b = pt.cast(max_run_b, "float64") / pt.cast(n_b, "float64")
    
    # Probability of choosing left (A)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
