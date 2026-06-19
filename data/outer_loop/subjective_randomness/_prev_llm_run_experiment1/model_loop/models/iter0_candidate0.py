"""
People judge a sequence as random based solely on how close its alternation rate
is to 50% — the rate expected from a fair coin. The sequence nearest to that
ideal alternation rate looks more random, regardless of H/T balance.

Refinement of prototype_similarity: retains the prototype-closeness mechanism
but uses only alternation rate (p_alts), dropping H/T balance entirely.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    tau = pm.HalfNormal("tau", sigma=5.0)

    deviation_a = pt.abs(p_alts_a - 0.5)
    deviation_b = pt.abs(p_alts_b - 0.5)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (deviation_b - deviation_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
