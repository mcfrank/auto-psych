"""
People evaluate the randomness of a sequence based on its proportion of tails, exhibiting a cognitive bias where tails are perceived as inherently more random than heads. Thus, when comparing two sequences, they are more likely to judge the sequence with a higher proportion of tails as the more random one.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    
    # Calculate proportion of tails
    p_tails_a = (n_a - h_a) / pt.clip(n_a, 1, 1000)
    p_tails_b = (n_b - h_b) / pt.clip(n_b, 1, 1000)
    
    # Free parameter: sensitivity to the difference in tails proportion.
    # A Normal prior allows the data to confirm the positive direction of the bias.
    weight = pm.Normal("weight", mu=0.0, sigma=10.0)
    
    # A positive difference means A has more tails than B
    # Since tails are perceived as more random, this should increase p_left
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(weight * (p_tails_a - p_tails_b)))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
