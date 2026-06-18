import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor

alpha = pt.scalar('alpha')
h = pt.scalar('h')
n = pt.scalar('n')

log_bias = pt.gammaln(h + alpha) + pt.gammaln(n - h + alpha) - pt.gammaln(n + 2*alpha) - (2 * pt.gammaln(alpha) - pt.gammaln(2*alpha))
log_fair = n * pt.log(0.5)

bf = log_fair - log_bias

f = pytensor.function([alpha, h, n], bf)
print(f(1.0, 5.0, 10.0))
print(f(1.0, 10.0, 10.0))
