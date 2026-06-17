"""
People judge a sequence as more random when its longest streak of identical
outcomes is shorter. A long run of repeated heads or tails is the single most
cognitively salient violation of randomness; sequences where the maximum run is
brief look random, and sequences containing even one long streak look
non-random. No other feature of the sequence matters once maximum run length is
accounted for.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))

    # Sensitivity to the difference in maximum run length.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Chose left (A) when A has the shorter maximum run.
    run_diff = max_run_norm_b - max_run_norm_a
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * run_diff))

    # Observed response.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
