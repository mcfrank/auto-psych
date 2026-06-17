import numpy as np
import pymc as pm
import pytensor.tensor as pt

"""
People judge the randomness of a sequence based solely on the overall balance of its outcomes. They perceive a sequence as less random the more its proportion of heads deviates from an even split, heavily penalizing any imbalance.
"""

with pm.Model() as model:
    # Stimulus inputs
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter for how strongly imbalance affects the judgment
    weight = pm.Normal("weight", mu=0.0, sigma=10.0)

    # Cognitive randomness score driven entirely by the sequence's imbalance
    score_a = weight * imbalance_a
    score_b = weight * imbalance_b

    # Choice probability using a logistic choice rule
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))
    
    # Clamp for numerical safety to avoid exactly 0 or 1
    p_left_safe = pt.clip(p_left, 1e-6, 1 - 1e-6)

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
