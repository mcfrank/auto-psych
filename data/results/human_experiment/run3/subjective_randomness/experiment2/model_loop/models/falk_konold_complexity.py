"""People judge the randomness of a sequence by its structural complexity when parsed into continuous alternating and repeating sub-sequences (Falk & Konold's difficulty of encoding), perceiving sequences with a lower rate of sub-sequences as simpler and therefore less random."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Feature inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))

    rep_motifs_a = pm.Data("rep_motifs_a", np.zeros(1, dtype="int64"))
    alt_motifs_a = pm.Data("alt_motifs_a", np.zeros(1, dtype="int64"))

    rep_motifs_b = pm.Data("rep_motifs_b", np.zeros(1, dtype="int64"))
    alt_motifs_b = pm.Data("alt_motifs_b", np.zeros(1, dtype="int64"))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Cognitive parameters
    beta = pm.Uniform("beta", lower=0.1, upper=20.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Complexity is the total number of chunks (Falk & Konold motifs)
    chunks_a = pt.cast(rep_motifs_a + alt_motifs_a, "float64")
    chunks_b = pt.cast(rep_motifs_b + alt_motifs_b, "float64")

    # We normalize by sequence length to get a chunk rate
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")

    complexity_a = chunks_a / n_a_f
    complexity_b = chunks_b / n_b_f

    # Higher complexity = more random
    score_a = complexity_a
    score_b = complexity_b

    # Softmax decision rule
    p_left_raw = pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    pm.Bernoulli("response", p=p_left, observed=chose_left)
