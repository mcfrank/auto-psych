"""
People judge a sequence as more random when its head count is more probable
under a fair coin: they evaluate the binomial likelihood B(h; n, 0.5) and
choose whichever sequence has a head count closer to the most probable outcome
for its length. This differs from linear imbalance — it encodes the full
combinatorial structure of how likely a given count is.
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

    # Log binomial coefficient: log C(n, h) = lgamma(n+1) - lgamma(h+1) - lgamma(n-h+1)
    # Higher log C(n, h) ↔ head count more probable under fair coin → sequence looks more random
    log_binom_a = pt.gammaln(n_a + 1) - pt.gammaln(h_a + 1) - pt.gammaln(n_a - h_a + 1)
    log_binom_b = pt.gammaln(n_b + 1) - pt.gammaln(h_b + 1) - pt.gammaln(n_b - h_b + 1)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (log_binom_a - log_binom_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
