import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    # Sensitivity to length-normalized max run difference.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Longer sequences naturally have longer max runs; normalize by sequence length
    # so that equal-"randomness" sequences score the same regardless of length.
    norm_run_a = pt.cast(max_run_a, "float64") / pt.cast(n_a, "float64")
    norm_run_b = pt.cast(max_run_b, "float64") / pt.cast(n_b, "float64")

    # Prefer the sequence with the shorter length-corrected max run (more random-looking).
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (norm_run_b - norm_run_a)))

    # Observed response.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
