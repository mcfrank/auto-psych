"""
People judge a sequence as more random when it contains a shorter maximum run —
the longest unbroken streak of the same outcome. Long streaks feel like a
pattern; the sequence whose worst streak is smallest looks most random,
regardless of overall balance or alternation rate.
"""
import numpy as np
import pymc as pm

with pm.Model() as model:
    # Normalized max run: (max_run - 1) / (n - 1), ranges 0 (no run) to 1 (all same).
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    tau = pm.HalfNormal("tau", sigma=5.0)

    # Sequence with shorter max run looks more random -> higher p_left when A has shorter run.
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (max_run_norm_b - max_run_norm_a)),
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
