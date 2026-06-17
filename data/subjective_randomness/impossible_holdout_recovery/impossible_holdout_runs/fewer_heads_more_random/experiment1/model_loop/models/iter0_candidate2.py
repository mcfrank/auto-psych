"""
People judge the randomness of a sequence by its proportion of heads, holding a biased belief that sequences containing a lower proportion of heads (and thus more tails) are more representative of a random coin.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    # Initializing n_a and n_b to 1.0 to avoid division by zero during graph compilation.
    n_a = pm.Data("n_a", np.ones(1, dtype="float64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="float64"))
    n_b = pm.Data("n_b", np.ones(1, dtype="float64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter representing the sensitivity to the difference in head proportions.
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Proportion of heads
    prop_heads_a = h_a / n_a
    prop_heads_b = h_b / n_b

    # Sequences with FEWER heads are hypothesized to be judged as MORE random.
    # Therefore, we assign a higher score to lower proportions of heads.
    score_a = -prop_heads_a
    score_b = -prop_heads_b

    # Probability of choosing left (sequence A)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
