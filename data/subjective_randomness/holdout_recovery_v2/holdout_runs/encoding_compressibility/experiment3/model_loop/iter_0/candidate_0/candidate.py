"""
People judge a sequence as more random when its head count is most combinatorially
plausible under a fair coin — closest to the mode of the binomial distribution.
They compare sequences by log C(n, h), the log number of arrangements with that
head count, penalizing imbalance in proportion to its combinatorial implausibility
rather than its raw magnitude.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    tau = pm.HalfNormal("tau", sigma=2.0)

    n_a_f = pt.cast(n_a, "float64")
    h_a_f = pt.cast(h_a, "float64")
    n_b_f = pt.cast(n_b, "float64")
    h_b_f = pt.cast(h_b, "float64")

    # log C(n, h) = log(n!) - log(h!) - log((n-h)!) via log-gamma
    log_binom_a = pt.gammaln(n_a_f + 1) - pt.gammaln(h_a_f + 1) - pt.gammaln(n_a_f - h_a_f + 1)
    log_binom_b = pt.gammaln(n_b_f + 1) - pt.gammaln(h_b_f + 1) - pt.gammaln(n_b_f - h_b_f + 1)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (log_binom_a - log_binom_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
