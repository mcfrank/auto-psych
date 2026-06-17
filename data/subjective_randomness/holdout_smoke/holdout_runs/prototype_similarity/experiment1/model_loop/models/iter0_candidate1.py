"""
People evaluate the randomness of a coin flip sequence based on its alternation rate.
Sequences with a higher proportion of alternations between outcomes are perceived as
more random due to an expectation that random processes switch frequently.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: proportion of alternations
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free parameter: sensitivity to the difference in alternation rates
    tau = pm.Normal("tau", mu=0.0, sigma=10.0)

    # Probability of choosing the left sequence (A) as more random
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (p_alts_a - p_alts_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
