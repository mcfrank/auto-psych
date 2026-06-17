import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    pid = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    
    # We can use shape=100 safely.
    beta_mu = pm.Normal("beta_mu", mu=0.0, sigma=1.0)
    beta_sigma = pm.HalfNormal("beta_sigma", sigma=1.0)
    beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, shape=100)
    beta = pt.exp(beta_mu + beta_sigma * beta_offset)
    
    p_beta = beta[pid]
