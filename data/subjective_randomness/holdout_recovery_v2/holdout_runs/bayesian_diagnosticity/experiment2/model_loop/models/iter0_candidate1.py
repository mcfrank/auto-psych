"""
People judge a sequence as more random-looking when it contains a shorter maximum
consecutive run of the same outcome. A long streak of identical flips is the most
salient cue that a sequence is not random, so when comparing two sequences people
choose the one whose longest run is shorter, and the strength of that preference
scales with how different the two maximum runs are.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — normalized max run length for each sequence.
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Sensitivity to max-run difference.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Prefer the sequence with the shorter maximum run.
    # When max_run_norm_b > max_run_norm_a, sequence A looks more random → p_left > 0.5.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (max_run_norm_b - max_run_norm_a)))

    # Observed response.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
