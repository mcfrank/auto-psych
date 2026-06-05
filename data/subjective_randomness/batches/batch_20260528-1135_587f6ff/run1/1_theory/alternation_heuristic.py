# file: alternation_heuristic.py
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus features: alternation proportions for sequences A and B
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter: sensitivity to the difference in alternation rates.
    # Constrained to be positive to reflect a preference for *more* alternations.
    tau = pm.HalfNormal("tau", sigma=10.0)

    # The probability of choosing sequence A (left) increases as its alternation
    # proportion exceeds that of sequence B.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (p_alts_a - p_alts_b)))

    # Observed responses
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
