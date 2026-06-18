"""People judge a sequence as more random the closer its proportion of heads is to 50%."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    # distance from 0.5
    dist_a = pt.abs(p_a - 0.5)
    dist_b = pt.abs(p_b - 0.5)
    
    # prefer smaller distance => negative weight
    # probability of choosing a is higher when dist_a < dist_b
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dist_b - dist_a)))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
