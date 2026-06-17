"""
People judge a sequence as more random-looking when its alternation rate is
closer to 0.5, the rate a fair coin naturally produces. They carry a fixed
internal prototype at exactly 0.5 — not a learned or context-dependent ideal —
and their preference is driven by linear distance from that standard: whichever
sequence's alternation rate deviates less from 0.5 looks more random, scaled
by a single sensitivity parameter.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    tau = pm.HalfNormal("tau", sigma=3.0)

    dev_a = pt.abs(p_alts_a - 0.5)
    dev_b = pt.abs(p_alts_b - 0.5)

    # When B deviates more from 0.5 than A, p_left > 0.5 (A chosen as more random)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dev_b - dev_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
