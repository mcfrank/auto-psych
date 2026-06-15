import numpy as np
import pytensor.tensor as pt
import pymc as pm

with pm.Model() as m:
    x = pm.Data("x", np.array([2.0]))
    y = pt.gammaln(x)
print("gammaln available in pytensor.tensor!")
