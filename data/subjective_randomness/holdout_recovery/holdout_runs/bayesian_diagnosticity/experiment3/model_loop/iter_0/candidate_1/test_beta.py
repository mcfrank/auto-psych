import pymc as pm
import pytensor.tensor as pt
import numpy as np

with pm.Model() as model:
    c = pm.Uniform("c", 0.1, 10)
    print("Model created.")
