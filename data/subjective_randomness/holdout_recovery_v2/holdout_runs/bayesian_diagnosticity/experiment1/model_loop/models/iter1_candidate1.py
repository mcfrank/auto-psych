"""
People judge a sequence as random-looking based on how much periodic structure
it contains. A truly random sequence should have no detectable, regularly-repeating
pattern, so a sequence that cycles through outcomes in a predictable rhythm looks
non-random. When comparing two sequences, people choose the one with lower
periodicity as the more random-looking one.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Sensitivity to periodicity difference.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Lower periodicity → more random-looking.
    # When A has lower periodicity than B, (periodicity_b - periodicity_a) > 0
    # and p_left > 0.5, so the model correctly predicts choosing A (left).
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (periodicity_b - periodicity_a)),
    )

    # Observed response — pm.Data tensor passed directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
