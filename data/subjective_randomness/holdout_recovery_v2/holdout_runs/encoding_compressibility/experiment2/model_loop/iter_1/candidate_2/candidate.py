"""
People judge a sequence as more random when its alternation rate is closer to
their internal ideal for what randomness looks like. When comparing two
sequences, they pick whichever one's alternation rate deviates less from this
ideal. The ideal alternation rate is a free parameter — not fixed at 0.5 —
because people's intuitions are known to be biased toward over-alternation.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Ideal alternation rate: free parameter with a mild prior centered near 0.5
    ideal_alt = pm.Beta("ideal_alt", alpha=3.0, beta=3.0)
    # Sensitivity: how sharply probability shifts with deviation difference
    tau = pm.HalfNormal("tau", sigma=5.0)

    dev_a = pt.abs(p_alts_a - ideal_alt)
    dev_b = pt.abs(p_alts_b - ideal_alt)

    # Sequence A wins when it deviates less from the ideal (dev_b > dev_a)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dev_b - dev_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
