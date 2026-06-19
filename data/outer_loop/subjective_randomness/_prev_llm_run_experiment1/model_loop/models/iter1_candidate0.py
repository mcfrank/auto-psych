"""
Refinement of prototype_similarity using a multiplicative (conjunctive) combination rule.

People judge a sequence as more random when it simultaneously satisfies both
fair-coin prototype criteria: alternation rate ≈ 50% AND head proportion ≈ 50%.
Unlike additive prototype distance, the two proximity scores are multiplied, so
violating either criterion sharply degrades the randomness impression even if the
other criterion is met perfectly.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Alternation rate (0 = all repeats, 1 = all alternations, ideal = 0.5)
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Imbalance = 2*|p_heads - 0.5|, ranges [0, 1], 0 = perfectly balanced
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Sensitivity parameter
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Proximity to prototype on each dimension, each in [0, 1]
    # alt_prox = 1 when p_alts = 0.5, falls to 0 at extremes (0 or 1)
    alt_prox_a = 1.0 - 2.0 * pt.abs(p_alts_a - 0.5)
    alt_prox_b = 1.0 - 2.0 * pt.abs(p_alts_b - 0.5)

    # bal_prox = 1 when perfectly balanced (imbalance=0), 0 when maximally unbalanced
    bal_prox_a = 1.0 - imbalance_a
    bal_prox_b = 1.0 - imbalance_b

    # Conjunctive prototype score: must satisfy both criteria together
    proto_score_a = alt_prox_a * bal_prox_a
    proto_score_b = alt_prox_b * bal_prox_b

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (proto_score_a - proto_score_b))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
