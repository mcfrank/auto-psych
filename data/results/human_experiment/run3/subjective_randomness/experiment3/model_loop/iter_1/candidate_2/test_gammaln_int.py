import pytensor.tensor as pt
import pymc as pm
import numpy as np

with pm.Model() as m:
    x = pt.as_tensor_variable(np.array([1, 2], dtype='int64'))
    alpha = pm.Exponential('alpha', lam=1.0) + 1e-4
    y = pt.gammaln(x + alpha)
    print(m.compile_logp()({"alpha_log__": 0.0}))
