import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs.
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=2.0)

    # Approximate log-probability of seeing a max run this long under a fair coin:
    # P(max run >= k in n flips) ~ n * 2^(1-k), so log P ~ log(n) - (k-1)*log(2).
    # Higher value = less surprising run = more random-looking sequence.
    log_surp_a = pt.log(pt.cast(n_a, "float64")) - (
        pt.cast(max_run_a, "float64") - 1.0
    ) * pt.log(2.0)
    log_surp_b = pt.log(pt.cast(n_b, "float64")) - (
        pt.cast(max_run_b, "float64") - 1.0
    ) * pt.log(2.0)

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (log_surp_a - log_surp_b))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
