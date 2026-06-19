import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor

x = pt.scalar('x')
y = pt.gammaln(x)
f = pytensor.function([x], y)
print(f(5.0))
