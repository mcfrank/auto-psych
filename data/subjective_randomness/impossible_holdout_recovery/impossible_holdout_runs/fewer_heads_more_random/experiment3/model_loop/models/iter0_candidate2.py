"""
People judge the randomness of a sequence strictly by the absolute number of heads it contains, preferring sequences with fewer heads. However, their choices include a baseline rate of random guessing due to occasional attentional lapses.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: absolute count of heads
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=10.0)
    # Lapse rate: proportion of trials where the participant guesses randomly (0.5)
    # Beta(1, 9) prior favors lower lapse rates (mean 0.1)
    epsilon = pm.Beta("epsilon", alpha=1.0, beta=9.0)

    # Sequences with FEWER heads are judged as MORE random.
    # Therefore, we assign a higher score to lower absolute counts of heads.
    score_a = -tau * pt.cast(h_a, "float64")
    score_b = -tau * pt.cast(h_b, "float64")

    # Probability of choosing left based purely on the head count penalty
    p_left_cognitive = pm.math.sigmoid(score_a - score_b)
    
    # Final probability includes a chance of random guessing
    p_left_raw = (1.0 - epsilon) * p_left_cognitive + epsilon * 0.5

    # Clip to ensure numerical stability
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
