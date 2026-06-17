"""
People evaluate the randomness of a sequence based solely on the number of heads it contains, but their perception of randomness scales as a power-law function of the head count rather than strictly linearly or quadratically. The model infers this exponent to capture exactly how marginal increases in head counts shape judgments.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=1.0)
    gamma = pm.HalfNormal("gamma", sigma=2.0)

    # Cast to float for safe exponentiation
    h_a_f = pt.cast(h_a, "float64")
    h_b_f = pt.cast(h_b, "float64")

    # Power-law values (adding small constant to avoid 0**gamma issues if h=0 and gamma < 1)
    val_a = pt.power(h_a_f + 1e-6, gamma)
    val_b = pt.power(h_b_f + 1e-6, gamma)

    # Compute probability of choosing left
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (val_a - val_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
