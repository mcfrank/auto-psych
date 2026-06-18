"""
Observers evaluate the randomness of a binary sequence using a simple directional heuristic based on alternations. Rather than penalizing deviations from a specific ideal rate, they judge a sequence as more random simply if it contains a higher proportion of alternations.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: proportion of alternations
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    
    # Sensitivity to the proportion of alternations
    # Positive prior implies higher proportion of alternations -> higher perceived randomness -> higher score
    beta = pm.HalfNormal("beta", sigma=2.0)
    
    # The score for each sequence is simply proportional to its proportion of alternations
    score_a = beta * p_alts_a
    score_b = beta * p_alts_b
    
    # Probability of choosing left (sequence a)
    # Uses sigmoid of the difference in scores, clamped for numerical stability
    p_raw = pm.math.sigmoid(score_a - score_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_raw, 1e-6, 1 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
