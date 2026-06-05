import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — head proportion and alternation proportion for each sequence.
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Weights for zero-order and first-order deviation from 0.5.
    w_prop = pm.HalfNormal("w_prop", sigma=2.0)
    w_alt = pm.HalfNormal("w_alt", sigma=2.0)

    # "Non-randomness" score: distance from perfectly random on both statistics.
    # A truly random sequence has p_heads = 0.5 and p_alts = 0.5.
    struct_a = w_prop * pt.abs(p_a - 0.5) + w_alt * pt.abs(p_alts_a - 0.5)
    struct_b = w_prop * pt.abs(p_b - 0.5) + w_alt * pt.abs(p_alts_b - 0.5)

    # Prefer whichever sequence scores lower (more random-looking).
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(struct_b - struct_a))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
