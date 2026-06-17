"""
People judge sequence randomness based purely on outcome frequencies, paradoxically perceiving sequences with a greater imbalance between the number of heads and tails as more random. However, their choices are also influenced by a constant baseline spatial or presentation order preference (a side bias) that shifts their probability of choosing the first sequence.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    # tau controls sensitivity to the imbalance difference
    tau = pm.HalfNormal("tau", sigma=10.0)
    # side_bias controls the baseline preference for the left sequence
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=2.0)

    # Hypothesis: score is directly proportional to imbalance
    score_a = imbalance_a
    score_b = imbalance_b

    # Choice probability with side bias
    p_left_raw = pm.math.sigmoid(tau * (score_a - score_b) + side_bias)
    
    # Expose and clamp the probability for numerical safety
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1 - 1e-6))

    # Observed response: the pm.Data tensor is passed directly to observed=
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
