import pytensor.tensor as pt
import pytensor

x = pt.iscalar('x') # integer scalar
y = pt.gammaln(x + 1)
f = pytensor.function([x], y)
print("gammaln(3) =", f(2))
