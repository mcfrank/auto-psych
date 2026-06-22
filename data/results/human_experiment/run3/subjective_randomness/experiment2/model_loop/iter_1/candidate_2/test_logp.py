import pymc as pm
import pytensor.tensor as pt
import numpy as np

with pm.Model() as m:
    n_a = pm.Data("n_a", np.array([10], dtype="int64"))
    h_a = pm.Data("h_a", np.array([5], dtype="int64"))

    ideal_p = pm.Beta("ideal_p", alpha=2, beta=2)

    # Check if pm.logp works
    logp_val = pm.logp(pm.Binomial.dist(n=n_a, p=ideal_p), h_a)

    pm.Deterministic("test_logp", logp_val)

print("It works!")
