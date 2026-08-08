"""PyMC model for the single longest-streak avoidance hypothesis.

People judge a coin-toss sequence as random by detecting streaks of identical outcomes:
the longer a sequence's single longest run is (relative to its own length), the more it
looks engineered and therefore non-random. This one streak-detection mechanism drives the
choice, so people tend to prefer whichever of two sequences has the shorter longest run.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: normalized longest run = (longest streak) / (sequence length).
    # A higher value means a more suspiciously long single streak.
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Observed response: exact pm.Data tensor passed to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters of the streak detector.
    # sensitivity > 0: how strongly a longer streak signals non-randomness.
    sensitivity = pm.HalfNormal("sensitivity", sigma=3.0)
    # side_bias: a general base tendency toward one side regardless of structure.
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # Randomness "look" of each sequence is degraded by its longest streak.
    # A longer streak on the left reduces the chance of choosing left.
    streak_diff = max_run_norm_a - max_run_norm_b

    p_left_raw = pm.math.sigmoid(side_bias - sensitivity * streak_diff)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
