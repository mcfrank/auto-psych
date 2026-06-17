import numpy as np
import pymc as pm
import pytensor.tensor as pt

"""
People evaluate the randomness of a sequence strictly based on the absolute number of heads it contains. However, their choices are subject to a constant lapse rate, meaning they occasionally guess randomly regardless of the head counts, rather than perfectly following a logistic choice curve.
"""

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=5.0)
    lapse = pm.Beta("lapse", alpha=1.0, beta=9.0)

    # Cognitive mechanism: absolute number of heads
    diff = pt.cast(h_a - h_b, "float64")
    
    # Logistic choice probability with lapse rate integration
    p_deterministic = pm.math.sigmoid(tau * diff)
    p_lapsed = lapse * 0.5 + (1.0 - lapse) * p_deterministic
    
    # Numerically safe bounded probability
    p_left = pm.Deterministic("p_left", pt.clip(p_lapsed, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
