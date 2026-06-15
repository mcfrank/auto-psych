"""
People judge a sequence as more random when its proportion of heads is closer to 0.5.
A perfectly balanced sequence looks like the output of a fair coin; a skewed sequence
looks biased. When choosing between two sequences, people pick the one whose proportion
of heads is nearer to perfect balance.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — imbalance = |p - 0.5|, already computed in the CSV.
    imbalance_a = pm.Data("imbalance_a", np.zeros(1))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1))

    # Sensitivity: how sharply people discriminate imbalance differences.
    tau = pm.HalfNormal("tau", sigma=5.0)

    # People prefer the less imbalanced (more balanced) sequence.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (imbalance_b - imbalance_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
