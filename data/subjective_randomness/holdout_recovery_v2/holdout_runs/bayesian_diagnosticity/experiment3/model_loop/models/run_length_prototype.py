"""People judge sequences by how close the maximum run length is to an internal
prototype for a random sequence — both too-long streaks and too-short maximum runs
(forced alternation) look non-random, and sequences whose longest run matches the
ideal are judged most random."""

import numpy as np
import pymc as pm

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Ideal normalized max run for a random-looking sequence (~1/3 of length)
    theta_run = pm.Beta("theta_run", alpha=2.0, beta=3.0)
    beta = pm.HalfNormal("beta", sigma=5.0)

    score_a = -beta * (max_run_norm_a - theta_run) ** 2
    score_b = -beta * (max_run_norm_b - theta_run) ** 2

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
