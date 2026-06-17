import numpy as np
import pytensor.tensor as pt
import pymc as pm

with pm.Model() as m:
    tau = pm.HalfNormal("tau", sigma=1.0)
    diff = pt.scalar("diff")
    p = 0.5 + pt.arctan(tau * diff) / np.pi
    pm.Bernoulli("y", p=p, observed=np.array([0, 1]))
print("success")
