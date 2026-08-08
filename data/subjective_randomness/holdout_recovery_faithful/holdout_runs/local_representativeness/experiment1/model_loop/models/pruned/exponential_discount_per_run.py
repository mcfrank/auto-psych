"""
Random-looking sequences are judged by an evidence-accumulation process where each distinct run provides a baseline quantum of evidence for randomness, but this total evidence is exponentially discounted by the sequence's absolute deviation from an inferred messy prototype (ideal balance and alternation rates), reflecting a sharp psychological generalization gradient where perceived randomness drops off multiplicatively as sequences violate prototype expectations.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    
    p_a = pm.Data("p_a", np.zeros(1, dtype="float64"))
    p_b = pm.Data("p_b", np.zeros(1, dtype="float64"))
    
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=1.0)
    ideal_p = pm.Beta("ideal_p", alpha=2.0, beta=2.0)
    ideal_alt = pm.Beta("ideal_alt", alpha=2.0, beta=2.0)
    
    w_imb = pm.HalfNormal("w_imb", sigma=5.0)
    w_alt = pm.HalfNormal("w_alt", sigma=5.0)

    # Number of runs is alternations + 1
    runs_a = alts_a + 1
    runs_b = alts_b + 1

    # Absolute deviations from prototype
    dev_imb_a = pt.abs(p_a - ideal_p)
    dev_alt_a = pt.abs(p_alts_a - ideal_alt)
    
    dev_imb_b = pt.abs(p_b - ideal_p)
    dev_alt_b = pt.abs(p_alts_b - ideal_alt)

    # Exponential discount of the baseline evidence (runs)
    discount_a = pt.exp(- (w_imb * dev_imb_a + w_alt * dev_alt_a))
    discount_b = pt.exp(- (w_imb * dev_imb_b + w_alt * dev_alt_b))

    score_a = runs_a * discount_a
    score_b = runs_b * discount_b

    # Probability of choosing left (sequence A)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (score_a - score_b)))

    # Observed response: the pm.Data tensor is passed directly to observed=
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
