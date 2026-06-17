"""
People judge sequence randomness by quadratic proximity to a prototype with
balanced H/T and an ideal alternation rate, but the penalty is asymmetric:
sequences that are too streaky (below the ideal alternation rate) are penalized
more harshly than sequences equally far in the over-alternating direction.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    p_a = pm.Data("p_a", np.zeros(1))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1))
    p_b = pm.Data("p_b", np.zeros(1))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1))

    # Observed response.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters.
    ideal_alts = pm.Beta("ideal_alts", alpha=2, beta=2)       # prototype alternation rate
    tau = pm.HalfNormal("tau", sigma=5.0)                      # inverse temperature
    streak_bias = pm.HalfNormal("streak_bias", sigma=2.0)      # extra penalty for streakiness

    # Head-balance penalty: symmetric quadratic deviation from 0.5.
    head_dev_a = (p_a - 0.5) ** 2
    head_dev_b = (p_b - 0.5) ** 2

    # Alternation-rate deviation from prototype.
    alt_dev_a = p_alts_a - ideal_alts
    alt_dev_b = p_alts_b - ideal_alts

    # Asymmetric multiplier: streaky sequences (below ideal) get extra penalty.
    streak_mult_a = pt.switch(pt.lt(alt_dev_a, 0.0), 1.0 + streak_bias, 1.0)
    streak_mult_b = pt.switch(pt.lt(alt_dev_b, 0.0), 1.0 + streak_bias, 1.0)

    alt_penalty_a = streak_mult_a * alt_dev_a ** 2
    alt_penalty_b = streak_mult_b * alt_dev_b ** 2

    # Randomness score: higher = more random. Distance from prototype = less random.
    score_a = -(head_dev_a + alt_penalty_a)
    score_b = -(head_dev_b + alt_penalty_b)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
