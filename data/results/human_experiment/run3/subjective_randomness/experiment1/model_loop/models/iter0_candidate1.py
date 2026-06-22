import numpy as np
import pymc as pm
import pytensor.tensor as pt

"""
People judge the randomness of a sequence by evaluating the statistical typicality of its macroscopic features, computing the joint Binomial log-probability of observing its number of heads and alternations under their subjective expectations for a random process.
"""

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    p_heads = pm.Beta("p_heads", alpha=10.0, beta=10.0)
    p_alts = pm.Beta("p_alts", alpha=6.0, beta=4.0)
    tau = pm.HalfNormal("tau", sigma=1.0)
    
    p_heads_safe = pt.clip(p_heads, 1e-6, 1.0 - 1e-6)
    p_alts_safe = pt.clip(p_alts, 1e-6, 1.0 - 1e-6)

    def log_binom(k, n, p):
        k_f = pt.cast(k, "float64")
        n_f = pt.cast(n, "float64")
        comb = pt.gammaln(n_f + 1.0) - pt.gammaln(k_f + 1.0) - pt.gammaln(n_f - k_f + 1.0)
        return comb + k_f * pt.log(p) + (n_f - k_f) * pt.log(1.0 - p)
        
    log_typ_h_a = log_binom(h_a, n_a, p_heads_safe)
    log_typ_alts_a = log_binom(alts_a, n_a - 1, p_alts_safe)
    log_typ_a = log_typ_h_a + log_typ_alts_a
    
    log_typ_h_b = log_binom(h_b, n_b, p_heads_safe)
    log_typ_alts_b = log_binom(alts_b, n_b - 1, p_alts_safe)
    log_typ_b = log_typ_h_b + log_typ_alts_b
    
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (log_typ_a - log_typ_b)))
    p_left_safe = pt.clip(p_left, 1e-6, 1.0 - 1e-6)
    
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
