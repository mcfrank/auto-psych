"""
People evaluate a sequence's randomness by how statistically typical its number of alternations is. They compute the exact binomial probability of observing the sequence's number of alternations under a fair coin, and prefer the sequence whose alternation count is more mathematically probable to occur by chance.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter
    tau = pm.HalfNormal("tau", sigma=1.0)

    # Binomial log-probability of observing 'alts' given 'n-1' opportunities with p=0.5
    # log( (n-1)! / (alts! * (n-1-alts)!) * 0.5**(n-1) )
    log_p_a = pt.gammaln(n_a) - pt.gammaln(alts_a + 1) - pt.gammaln(n_a - alts_a) + (n_a - 1) * pt.log(0.5)
    log_p_b = pt.gammaln(n_b) - pt.gammaln(alts_b + 1) - pt.gammaln(n_b - alts_b) + (n_b - 1) * pt.log(0.5)
    
    # Preference for the sequence with the higher binomial probability
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (log_p_a - log_p_b)))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
