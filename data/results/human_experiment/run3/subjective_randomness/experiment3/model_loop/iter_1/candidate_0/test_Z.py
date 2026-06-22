import numpy as np
import pytensor.tensor as pt
import pytensor

n_f = pt.dvector('n_f')
gamma = pt.dscalar('gamma')

max_len = 100
k = pt.arange(max_len + 1, dtype="float64")
k_exp = pt.expand_dims(k, 1)
n_f_exp = pt.expand_dims(n_f, 0)

mask = pt.cast(k_exp <= n_f_exp, "float64")
unnormalized = pt.exp(-gamma * pt.abs(k_exp - n_f_exp / 2.0)) * mask
Z = pt.sum(unnormalized, axis=0)

f = pytensor.function([n_f, gamma], Z)
print("n=4, gamma=1.0:", f(np.array([4.0]), 1.0))
print("Expected n=4:", np.sum(np.exp(-1.0 * np.abs(np.arange(5) - 2.0))))
