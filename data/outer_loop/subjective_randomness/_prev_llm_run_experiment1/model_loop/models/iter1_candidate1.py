"""
People judge a sequence as more random when it contains fewer repetitive
sub-patterns — runs of identical outcomes of any length (e.g., 'HH', 'HHH',
'HHHH'). Each such repetitive motif is salient evidence that the sequence came
from a non-random process; the sequence with the lower total count of repetitive
motifs looks more random, regardless of overall balance or alternation rate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    rep_motifs_a = pm.Data("rep_motifs_a", np.zeros(1, dtype="int64"))
    rep_motifs_b = pm.Data("rep_motifs_b", np.zeros(1, dtype="int64"))

    # Sensitivity to differences in repetitive motif count.
    tau = pm.HalfNormal("tau", sigma=2.0)

    # Fewer rep_motifs → more random-looking.
    # p_left increases when sequence A has fewer repetitive motifs than B.
    diff = pt.cast(rep_motifs_b - rep_motifs_a, "float64")
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * diff))

    # Observed response.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
