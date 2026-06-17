"""
People judge the randomness of a sequence by comparing its maximum run proportion against a subjective tolerance limit. Sequences with a longest streak below this limit are perceived as perfectly acceptable, but any maximum run proportion exceeding the tolerance incurs a linearly increasing penalty for appearing too streaky.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters with priors.
    # tolerance: subjective maximum acceptable run proportion
    tolerance = pm.Beta("tolerance", alpha=2.0, beta=5.0)
    # tau: choice determinism
    tau = pm.HalfNormal("tau", sigma=10.0)

    # One-sided penalty (hinge loss): penalize only if max_run_norm exceeds tolerance.
    penalty_a = pt.maximum(0.0, max_run_norm_a - tolerance)
    penalty_b = pt.maximum(0.0, max_run_norm_b - tolerance)

    # Calculate choice probability.
    # If option B has a higher penalty (more streaky than tolerated),
    # option A is perceived as more random, making p_left > 0.5.
    diff = penalty_b - penalty_a
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * diff))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
