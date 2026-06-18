"""Refining the hypothesis that observers track the frequency of heads, we propose they evaluate subjective randomness using a simple directional heuristic. Rather than penalizing symmetric deviations from a 50% proportion, observers judge a sequence as more random simply if it contains a greater absolute number of heads."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    
    # Free cognitive parameter for decision noise/scaling
    tau = pm.HalfNormal("tau", sigma=5.0)
    
    # Score is simply the absolute number of heads
    score_a = pt.cast(h_a, "float64")
    score_b = pt.cast(h_b, "float64")
    
    # Decision rule: higher score (more heads) -> higher probability of choosing sequence as more random
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
