"""
People judge sequence randomness by evaluating the raw numerical difference between the counts of the two outcomes, perceiving sequences with a larger absolute difference between the number of heads and tails as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="float64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="float64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="float64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter
    weight = pm.Normal("weight", mu=0.0, sigma=10.0)

    # Raw count difference: |heads - tails| = |heads - (n - heads)| = |2 * heads - n|
    raw_diff_a = pt.abs(2.0 * h_a - n_a)
    raw_diff_b = pt.abs(2.0 * h_b - n_b)

    # Calculate probability of choosing left (A)
    score_a = weight * raw_diff_a
    score_b = weight * raw_diff_b
    
    p_left_raw = pm.math.sigmoid(score_a - score_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
