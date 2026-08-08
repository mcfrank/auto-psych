"""PyMC adapter for the Falk & Konold (1997) Difficulty Predictor family.

Randomness = DP = rep_motifs + 2*alt_motifs (the minimal-DP parse computed by
the featurizer), unnormalised by length; harder-to-encode sequences seem more
random. The theory has no free cognitive parameters — only the choice rule's
``beta`` and ``side_bias`` are inferred. See the pure-Python twin in
``model_families/falk_konold_dp.py`` for the full rationale.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    rep_motifs_a = pm.Data("rep_motifs_a", np.zeros(1, dtype="int64"))
    alt_motifs_a = pm.Data("alt_motifs_a", np.zeros(1, dtype="int64"))
    rep_motifs_b = pm.Data("rep_motifs_b", np.zeros(1, dtype="int64"))
    alt_motifs_b = pm.Data("alt_motifs_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = pt.cast(rep_motifs_a + 2 * alt_motifs_a, "float64")
    score_b = pt.cast(rep_motifs_b + 2 * alt_motifs_b, "float64")

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
