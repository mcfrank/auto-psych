"""
People judge a sequence as more random when its proportion of heads is closer
to 50% — the rate expected from a fair coin. The sequence whose head proportion
is nearest to 0.5 looks most random, regardless of run structure or alternation.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — head proportions for each sequence
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))

    # Sensitivity to the balance difference
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Distance from the fair-coin baseline (0.5); smaller = more random
    dist_a = pt.abs(p_a - 0.5)
    dist_b = pt.abs(p_b - 0.5)

    # Sequence A looks more random when its distance is smaller
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dist_b - dist_a)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
