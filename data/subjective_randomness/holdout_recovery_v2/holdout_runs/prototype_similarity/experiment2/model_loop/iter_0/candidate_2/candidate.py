"""
People judge a sequence as more random when its longest unbroken run of identical
outcomes is shorter. The maximum run length is the sole cue: a long streak signals
a non-random, streaky process, so people pick whichever sequence has the shorter
maximum run as looking more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))

    # Sensitivity: how strongly does run-length difference drive choice?
    tau = pm.HalfNormal("tau", sigma=2.0)

    # A feels more random when its max run is shorter (negative difference).
    run_diff = pt.cast(max_run_a - max_run_b, "floatX")
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(-tau * run_diff))

    # Observed response — pm.Data tensor passed directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
