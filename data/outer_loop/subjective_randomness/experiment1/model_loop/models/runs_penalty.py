"""Observers treat long unbroken runs (streaks) as a hallmark of non-randomness.
The sequence with the shorter maximum run is preferred as more random, implemented
as a softmax over the difference in max-run length. Captures the intuition that
patterns like HHHHHH or TTTTTT "feel" deterministic, not random.

Complements alternation_bias: both predict the same direction for extreme
sequences but can diverge on sequences with moderate alternation and no long runs."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Longest unbroken run in each sequence (integer count).
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=2.0)  # softmax temperature

    # Prefer A when max_run_b > max_run_a (A has shorter streak → more random).
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * pt.cast(max_run_b - max_run_a, "float64")),
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
