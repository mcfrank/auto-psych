import pymc as pm
import numpy as np

with pm.Model() as m:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))

    val = pm.logp(pm.Binomial.dist(n=n_a, p=0.5), h_a)
    pm.Deterministic("val", val)

print("Works for zeros!")
