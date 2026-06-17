import pytensor.tensor as pt
import numpy as np
import pytensor

x = pt.scalar('x')
y = pt.gammaln(x)
f = pytensor.function([x], y)
print("gammaln(1.0) =", f(1.0))
print("gammaln(2.0) =", f(2.0))
print("gammaln(3.0) =", f(3.0))
