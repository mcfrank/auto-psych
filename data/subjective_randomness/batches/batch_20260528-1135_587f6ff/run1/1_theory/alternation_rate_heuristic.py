# file: alternation_rate_heuristic.py
"""Observers judge randomness based on how close the sequence's alternation rate is to their subjective ideal alternation rate."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: alternation proportions for the two sequences
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters:
    # 1. Subjective ideal alternation rate (human data often shows a preference for ~0.6-0.7)
    ideal_alt = pm.Beta("ideal_alt", alpha=2.0, beta=2.0)
    # 2. Softmax temperature for the decision rule
    tau = pm.HalfNormal("tau", sigma=10.0)

    # The "randomness score" is the negative absolute distance to the ideal alternation rate
    score_a = -abs(p_alts_a - ideal_alt)
    score_b = -abs(p_alts_b - ideal_alt)

    # Probability of choosing sequence A (left)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    # Observed responses
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
