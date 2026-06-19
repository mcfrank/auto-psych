"""
Conjunctive prototype refinement of prototype_similarity.

People judge a sequence as random when it is near a mental prototype on both
alternation rate and head-balance simultaneously. The dimension with the larger
deviation from the prototype (Chebyshev / worst-case distance) governs the
judgment: failing either criterion badly is enough to make a sequence look
non-random, regardless of how well it scores on the other dimension.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Sensitivity: how strongly the prototype distance difference drives choice.
    tau = pm.HalfNormal("tau", sigma=5.0)

    # Conjunctive (Chebyshev) distance: the worse dimension governs.
    # Ideal alternation rate = 0.5; imbalance is already an unsigned magnitude.
    alt_dist_a = pt.abs(p_alts_a - 0.5)
    alt_dist_b = pt.abs(p_alts_b - 0.5)

    dist_a = pt.maximum(alt_dist_a, imbalance_a)
    dist_b = pt.maximum(alt_dist_b, imbalance_b)

    # Smaller prototype distance → looks more random → tends to be chosen.
    # dist_b > dist_a means A is closer to prototype → p_left increases.
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (dist_b - dist_a))
    )

    # Observed response — pm.Data tensor passed directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
