import pymc as pm
import pytensor.tensor as pt
import numpy as np

with pm.Model() as model:
    pid = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, shape=30)
    beta = pt.exp(beta_offset[pid])
    
    y = pm.Data("y", np.zeros(1, dtype="int64"))
    pm.Bernoulli("obs", p=pm.math.sigmoid(beta), observed=y)

print("Compiled successfully")
