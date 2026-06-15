"""
People judge randomness by detecting periodic (rhythmic) structure in sequences.
A sequence with higher periodicity is perceived as more patterned and less random.
On each trial, the person chooses whichever sequence has lower periodicity as the
more random-looking one.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Sensitivity to periodicity differences (inverse temperature).
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Left sequence chosen when it has lower periodicity than right sequence.
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (periodicity_b - periodicity_a)),
    )

    # Observed response: the pm.Data tensor passed directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
