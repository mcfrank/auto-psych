"""
People evaluate the randomness of a sequence based on its global, length-normalized properties rather than by accumulating evidence over time. They compute a penalty based on the squared deviation of the sequence's overall proportion of heads and alternation rate from an ideal mental prototype, penalizing extreme deviations more heavily, but evaluating this average typicality without multiplying it by the sequence length.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs for sequence A
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    
    # Stimulus inputs for sequence B
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    # Ideal prototypes for proportion of heads and alternation rate
    ideal_p = pm.Beta("ideal_p", alpha=5.0, beta=5.0)
    ideal_alts = pm.Beta("ideal_alts", alpha=5.0, beta=5.0)
    
    # Weights for the two feature deviations
    w_p = pm.HalfNormal("w_p", sigma=10.0)
    w_alts = pm.HalfNormal("w_alts", sigma=10.0)

    # Length-independent penalty (rate-based typicality)
    # Uses squared deviation from the prototype to heavily penalize extreme values
    penalty_a = w_p * ((p_a - ideal_p) ** 2) + w_alts * ((p_alts_a - ideal_alts) ** 2)
    penalty_b = w_p * ((p_b - ideal_p) ** 2) + w_alts * ((p_alts_b - ideal_alts) ** 2)
    
    # Probability of choosing sequence A (left). 
    # Sequence A is more likely to be chosen if its penalty is lower than sequence B's penalty.
    p_left_raw = pm.math.sigmoid(penalty_b - penalty_a)
    
    # Clamp for numerical safety
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
