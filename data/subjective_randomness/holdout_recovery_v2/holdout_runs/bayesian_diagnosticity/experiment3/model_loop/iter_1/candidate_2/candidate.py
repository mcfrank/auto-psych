"""
People judge a sequence as more random-looking when its alternation rate is closer to 0.5,
the rate a fair coin produces. Their internal prototype is fixed at exactly 0.5 — not a
learned parameter — so the only free parameter is sensitivity to deviations from that fixed
standard.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    tau = pm.HalfNormal("tau", sigma=5.0)

    dev_a = pt.abs(p_alts_a - 0.5)
    dev_b = pt.abs(p_alts_b - 0.5)

    # Sequence A looks more random when it deviates less from 0.5 than B does.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dev_b - dev_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
