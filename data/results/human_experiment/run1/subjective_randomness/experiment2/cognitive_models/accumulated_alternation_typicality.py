"""
People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this typicality is based entirely on how closely the sequence's alternation rate matches a mental prototype, completely ignoring the overall proportion of heads and tails.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    ideal_alt = pm.Beta("ideal_alt", alpha=2.0, beta=2.0)
    w_alt = pm.HalfNormal("w_alt", sigma=5.0)
    base_typ = pm.Normal("base_typ", mu=0.0, sigma=5.0)

    # Calculate empirical rates (safeguard against division by zero)
    alt_rate_a = alts_a / pt.maximum(n_a - 1, 1)
    alt_rate_b = alts_b / pt.maximum(n_b - 1, 1)

    # Calculate per-event typicality as a base rate minus the deviation penalty
    typ_a = base_typ - w_alt * pt.square(alt_rate_a - ideal_alt)
    typ_b = base_typ - w_alt * pt.square(alt_rate_b - ideal_alt)

    # Total randomness score is the accumulated typicality over the sequence length
    rand_a = n_a * typ_a
    rand_b = n_b * typ_b

    # Choice probability
    p_left_raw = pm.math.sigmoid(rand_a - rand_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
