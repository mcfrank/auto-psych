import numpy as np
import pymc as pm
import pytensor.tensor as pt

"""
People judge sequence randomness based on outcome frequencies, perceiving sequences with a larger absolute difference between the raw count of heads and tails as more random.
"""

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter for decision noise / scaling
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Calculate absolute difference between count of heads and count of tails
    t_a = n_a - h_a
    t_b = n_b - h_b
    
    # pt.cast is used to ensure float64 for multiplication with tau
    diff_a = pt.cast(pt.abs(h_a - t_a), "float64")
    diff_b = pt.cast(pt.abs(h_b - t_b), "float64")

    # Greater difference -> perceived as more random
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (diff_a - diff_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
