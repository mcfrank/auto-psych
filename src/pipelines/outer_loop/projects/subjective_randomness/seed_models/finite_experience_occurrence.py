"""PyMC adapter for the Hahn & Warren (2009) finite-experience family.

Randomness is the log probability that the sequence occurs at least once in
the paper's focal finite stream of 20 fair flips. The occurrence probability
is precomputed exactly by the featurizer. The source analysis compares only
equal-length strings, so an explicit graph assertion rejects cross-length data
rather than silently introducing a new theory of length comparison.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.raise_op import Assert

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    occ_n20_a = pm.Data("occ_n20_a", np.zeros(1, dtype="float64"))
    occ_n20_b = pm.Data("occ_n20_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    checked_n_a = Assert(
        "finite_experience_occurrence requires same-length alternatives"
    )(n_a, pt.all(pt.eq(n_a, n_b)))

    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    length_check = pt.sum(pt.cast(checked_n_a - n_a, "float64"))
    score_a = pm.math.log(occ_n20_a) + length_check
    score_b = pm.math.log(occ_n20_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
