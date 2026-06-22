import pymc as pm
import numpy as np

with pm.Model() as m:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    x = pm.math.maximum(1, n_a - 1)
    pm.Deterministic("x", x)

print("Works!")
