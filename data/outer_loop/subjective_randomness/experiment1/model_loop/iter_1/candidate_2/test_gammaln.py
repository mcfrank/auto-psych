import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as m:
    n = pm.Data("n", np.array([4]))
    alts = pm.Data("alts", np.array([2]))
    val = pt.gammaln(n) - pt.gammaln(alts + 1) - pt.gammaln(n - alts)
    
print(val.eval())
