"""
People judge a sequence as more random when its head proportion is closer to 0.5,
but with a quadratic sensitivity to imbalance: each additional unit of deviation
from 50% heads incurs a disproportionately larger randomness penalty, so mild
imbalance is nearly ignored while extreme imbalance is strongly penalized. This
is a pure balance-checking mechanism with an accelerating discrimination curve,
differing from the linear head_balance model in how imbalance is psychologically scaled.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    tau = pm.HalfNormal("tau", sigma=5.0)

    # Quadratic penalty: squared deviation from 0.5 amplifies sensitivity
    # to extreme imbalance relative to the linear (head_balance) form.
    score_a = -(imbalance_a ** 2)
    score_b = -(imbalance_b ** 2)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
