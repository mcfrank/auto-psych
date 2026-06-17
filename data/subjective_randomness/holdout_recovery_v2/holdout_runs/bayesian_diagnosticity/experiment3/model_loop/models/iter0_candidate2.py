"""
People judge a sequence as more random-looking when its outcomes are more
evenly balanced between heads and tails. The single cognitive mechanism is
outcome balance: a sequence where heads and tails are equal looks maximally
random, and any departure from that balance reduces perceived randomness.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — imbalance = |proportion_heads - 0.5|, higher = less balanced.
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Free parameter: sensitivity to imbalance differences.
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Sequence A looks more random when it has lower imbalance than B.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (imbalance_b - imbalance_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
