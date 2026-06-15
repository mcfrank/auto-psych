import numpy as np
import pymc as pm

# Minimal run-aversion model: people judge randomness based solely on the
# longest run in the sequence, normalized by length. Two parameters only
# (w_run, side_bias). Tests whether periodicity and imbalance in inner_loop_model
# earn their keep, or whether max_run_norm alone captures participant choices.

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    w_run = pm.StudentT("w_run", nu=4.0, mu=0.0, sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(w_run * (max_run_norm_a - max_run_norm_b) + side_bias),
    )

    pm.Bernoulli("response", p=p_left, observed=chose_left)
