"""
Outcome balance hypothesis: people choose the sequence whose proportion of heads
and tails is closest to 50/50 as more random. Outcome balance is the single
salient cue — a fair coin should produce roughly equal numbers of each outcome,
so a sequence that deviates from balance looks non-random regardless of its
alternation rate or streak structure.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: pre-computed imbalance = |p_heads - 0.5| * 2, one per sequence.
    imbalance_a = pm.Data("imbalance_a", np.zeros(1))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1))

    # Sensitivity: how strongly imbalance difference drives the choice.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Sequence A looks more random when its imbalance is smaller.
    # Positive score_diff → prefer A (left) → p_left > 0.5.
    score_diff = imbalance_b - imbalance_a
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * score_diff))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
