"""
People judge a sequence as more random when its longest consecutive run is
close to the expected maximum run for a fair coin of that length (~log2(n)).
A max run that is too long signals a streaky process; one that is too short
signals an artificially regular process. Perceived randomness peaks at the
characteristic max-run and declines quadratically in either direction.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    # Sensitivity: how steeply randomness falls off from the ideal max run.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Expected longest run for a fair coin of length n ≈ log2(n).
    expected_a = pt.log(pt.cast(n_a, "float64")) / pt.log(2.0)
    expected_b = pt.log(pt.cast(n_b, "float64")) / pt.log(2.0)

    # Quadratic penalty around the expected max run.
    score_a = -(pt.cast(max_run_a, "float64") - expected_a) ** 2
    score_b = -(pt.cast(max_run_b, "float64") - expected_b) ** 2

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
