"""People judge a sequence as more random if its proportion of alternations is closer to their subjective ideal alternation rate."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    
    ideal_rate = pm.Beta("ideal_rate", alpha=2.0, beta=2.0)
    tau = pm.HalfNormal("tau", sigma=10.0)
    
    dist_a = pt.abs(p_alts_a - ideal_rate)
    dist_b = pt.abs(p_alts_b - ideal_rate)
    
    # prefer smaller distance => probability of choosing a is higher when dist_a < dist_b
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dist_b - dist_a)))
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
