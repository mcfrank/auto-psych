"""
People judge a sequence as more random when it has shorter maximum runs. A long
run (e.g., HHHHH) can be compactly described via run-length encoding — just name
the element and its length — making the sequence feel structured and non-random.
The sequence whose longest run is shorter resists this compact description and
therefore appears more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Sensitivity to run-length compressibility difference.
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Sequence with higher max_run_norm is more compressible → more structured.
    # p_left increases when max_run_norm_b > max_run_norm_a (B is more structured).
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (max_run_norm_b - max_run_norm_a)),
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
