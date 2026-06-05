import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Head proportions for each sequence.
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    # Sensitivity to fairness deviation.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Prefer the sequence whose proportion is closer to 0.5 (more "fair-looking").
    dev_a = pt.abs(p_a - 0.5)
    dev_b = pt.abs(p_b - 0.5)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dev_b - dev_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
