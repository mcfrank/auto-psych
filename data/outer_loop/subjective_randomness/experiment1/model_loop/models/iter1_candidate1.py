"""
People judge which sequence is more random by computing the Falk-Konold Difficulty
Predictor (DP = rep_motifs + 2 * alt_motifs). The sequence requiring more motifs to
describe under minimal run-based encoding appears more random, because it resists
compact description — purely streaky and purely alternating sequences both have low
DP and are rejected, while sequences with unpredictable mixtures have high DP and
appear random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    rep_motifs_a = pm.Data("rep_motifs_a", np.zeros(1, dtype="int64"))
    alt_motifs_a = pm.Data("alt_motifs_a", np.zeros(1, dtype="int64"))
    rep_motifs_b = pm.Data("rep_motifs_b", np.zeros(1, dtype="int64"))
    alt_motifs_b = pm.Data("alt_motifs_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=2.0)

    dp_a = pt.cast(rep_motifs_a, "float64") + 2.0 * pt.cast(alt_motifs_a, "float64")
    dp_b = pt.cast(rep_motifs_b, "float64") + 2.0 * pt.cast(alt_motifs_b, "float64")

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dp_a - dp_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
