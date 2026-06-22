"""Random-looking sequences are those that are cognitively harder to encode into chunks, as quantified by Falk and Konold's Difficulty Predictor (the number of repetition motifs plus twice the number of alternation motifs) normalized by sequence length."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    rep_motifs_a = pm.Data("rep_motifs_a", np.zeros(1, dtype="int64"))
    rep_motifs_b = pm.Data("rep_motifs_b", np.zeros(1, dtype="int64"))
    alt_motifs_a = pm.Data("alt_motifs_a", np.zeros(1, dtype="int64"))
    alt_motifs_b = pm.Data("alt_motifs_b", np.zeros(1, dtype="int64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free parameters
    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # Convert to float
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")
    rep_a_f = pt.cast(rep_motifs_a, "float64")
    rep_b_f = pt.cast(rep_motifs_b, "float64")
    alt_a_f = pt.cast(alt_motifs_a, "float64")
    alt_b_f = pt.cast(alt_motifs_b, "float64")

    # Difficulty Predictor (DP) = n1 + 2*n2
    dp_a = rep_a_f + 2.0 * alt_a_f
    dp_b = rep_b_f + 2.0 * alt_b_f

    # Normalize by sequence length (guard against div by zero, though n >= 2 in this task)
    dp_norm_a = dp_a / pt.clip(n_a_f, 1.0, np.inf)
    dp_norm_b = dp_b / pt.clip(n_b_f, 1.0, np.inf)

    # Higher difficulty -> more random -> higher score
    score_a = dp_norm_a
    score_b = dp_norm_b

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )

    pm.Bernoulli("response", p=p_left, observed=chose_left)
