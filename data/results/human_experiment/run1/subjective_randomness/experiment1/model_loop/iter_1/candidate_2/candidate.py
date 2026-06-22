"""
People judge the randomness of a sequence by comparing it to a mental prototype, but instead of focusing on global alternation rates, they evaluate the local 'clumpiness' of the sequence via its density of repeated motifs. Sequences are perceived as more random when their proportion of heads and their proportion of repeated motifs have a smaller squared deviation from a subjectively biased ideal.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Features from responses
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    rep_motifs_a = pm.Data("rep_motifs_a", np.zeros(1, dtype="int64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    rep_motifs_b = pm.Data("rep_motifs_b", np.zeros(1, dtype="int64"))

    # Free parameters for the subjective prototype
    ideal_p = pm.Beta("ideal_p", alpha=2.0, beta=2.0)
    ideal_rep = pm.Beta("ideal_rep", alpha=2.0, beta=2.0)
    
    # Weights for the deviations
    w_p = pm.HalfNormal("w_p", sigma=5.0)
    w_rep = pm.HalfNormal("w_rep", sigma=5.0)
    
    # Calculate empirical rates (safeguard against division by zero)
    p_a = pt.cast(h_a, "float64") / pt.maximum(pt.cast(n_a, "float64"), 1.0)
    p_b = pt.cast(h_b, "float64") / pt.maximum(pt.cast(n_b, "float64"), 1.0)
    
    rep_rate_a = pt.cast(rep_motifs_a, "float64") / pt.maximum(pt.cast(n_a, "float64"), 1.0)
    rep_rate_b = pt.cast(rep_motifs_b, "float64") / pt.maximum(pt.cast(n_b, "float64"), 1.0)
    
    # Calculate squared deviations from the subjective ideals
    dev_a = w_p * pt.square(p_a - ideal_p) + w_rep * pt.square(rep_rate_a - ideal_rep)
    dev_b = w_p * pt.square(p_b - ideal_p) + w_rep * pt.square(rep_rate_b - ideal_rep)
    
    # Lower deviation means it is closer to the prototype (more random)
    p_left_raw = pm.math.sigmoid(dev_b - dev_a)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))
    
    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
