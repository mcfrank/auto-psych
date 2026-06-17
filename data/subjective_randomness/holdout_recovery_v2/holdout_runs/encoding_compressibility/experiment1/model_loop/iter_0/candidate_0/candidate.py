"""
People choose the sequence that is harder to compress. A sequence with high
periodicity can be described compactly as a repeating pattern, which makes it
feel structured and non-random. The sequence with lower periodicity resists
this kind of compact description and therefore appears more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Sensitivity to periodicity difference; must be positive so lower
    # periodicity → more random-looking.
    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0, sigma=1)

    # When B is more periodic (more compressible) than A, score is positive
    # → p_left > 0.5 → prefer A as the more random-looking sequence.
    score = beta * (periodicity_b - periodicity_a) + side_bias
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
