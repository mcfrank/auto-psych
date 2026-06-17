import pytensor.tensor as pt
import pymc as pm
import numpy as np

try:
    print("pt.gammaln:", hasattr(pt, "gammaln"))
except: pass
try:
    print("pm.math.gammaln:", hasattr(pm.math, "gammaln"))
except: pass
