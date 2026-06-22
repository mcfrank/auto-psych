import pytensor.tensor as pt
import pymc as pm
import numpy as np

with pm.Model() as m:
    x = pt.as_tensor_variable(np.array([1.0, 2.0]))
    y = pt.gammaln(x)
    print(y.eval())
