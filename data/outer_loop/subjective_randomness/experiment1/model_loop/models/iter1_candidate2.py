"""
People judge which sequence looks more random by detecting periodic structure.
Regular, cyclic patterns signal a non-random generator; the sequence with lower
periodicity is chosen as more random. No other feature — balance, run length,
or alternation rate — influences the judgment.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — periodicity of each sequence (higher = more cyclic/patterned)
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))

    # Sensitivity: how strongly periodicity difference drives the choice
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Lower periodicity → looks more random → more likely to be chosen
    # p_left = P(chose A | A has lower periodicity than B)
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (periodicity_b - periodicity_a))
    )

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
