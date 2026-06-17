"""
People judge a sequence as more random-looking when its alternation rate is closer
to 0.5 — the rate a fair coin produces. The prototype is fixed at exactly 0.5
(not a learned parameter); only response sensitivity (tau) is free. The sequence
whose switching rate deviates less from 0.5 is chosen as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    p_alts_a = pm.Data("p_alts_a", np.zeros(1))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1))

    # Response sensitivity: how sharply deviations from 0.5 drive choices.
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Absolute deviation from the fixed fair-coin prototype (0.5).
    dev_a = pt.abs(p_alts_a - 0.5)
    dev_b = pt.abs(p_alts_b - 0.5)

    # Sequence with smaller deviation looks more random → more likely chosen left.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dev_b - dev_a)))

    # Observed response: pm.Data tensor passed directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
