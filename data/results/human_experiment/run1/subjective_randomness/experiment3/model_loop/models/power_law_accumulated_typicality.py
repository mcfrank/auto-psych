"""
People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but this evidence integration exhibits diminishing returns. Rather than growing linearly, the accumulated typicality scales according to a power-law function of sequence length, preventing extremely long sequences from disproportionately dominating judgments and explaining why the preference for longer sequences saturates.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    ideal_p = pm.Beta("ideal_p", alpha=2.0, beta=2.0)
    ideal_alt = pm.Beta("ideal_alt", alpha=2.0, beta=2.0)

    # Weights for the deviations
    w_p = pm.HalfNormal("w_p", sigma=5.0)
    w_alt = pm.HalfNormal("w_alt", sigma=5.0)

    # Base typicality per event
    base_typ = pm.Normal("base_typ", mu=0.0, sigma=5.0)

    # Power law exponent for diminishing returns on length
    # Beta prior naturally restricts it to (0, 1) capturing the "diminishing returns" hypothesis
    length_power = pm.Beta("length_power", alpha=2.0, beta=2.0)

    # Calculate empirical rates (safeguard against division by zero)
    n_a_f = pt.cast(pt.maximum(n_a, 1), "float64")
    n_b_f = pt.cast(pt.maximum(n_b, 1), "float64")

    p_a = pt.cast(h_a, "float64") / n_a_f
    p_b = pt.cast(h_b, "float64") / n_b_f

    alt_rate_a = pt.cast(alts_a, "float64") / pt.maximum(
        pt.cast(n_a - 1, "float64"), 1.0
    )
    alt_rate_b = pt.cast(alts_b, "float64") / pt.maximum(
        pt.cast(n_b - 1, "float64"), 1.0
    )

    # Calculate per-event typicality
    typ_a = base_typ - (
        w_p * pt.square(p_a - ideal_p) + w_alt * pt.square(alt_rate_a - ideal_alt)
    )
    typ_b = base_typ - (
        w_p * pt.square(p_b - ideal_p) + w_alt * pt.square(alt_rate_b - ideal_alt)
    )

    # Total randomness score uses a power-law function of sequence length
    rand_a = pt.pow(pt.cast(n_a, "float64"), length_power) * typ_a
    rand_b = pt.pow(pt.cast(n_b, "float64"), length_power) * typ_b

    # Choice probability
    p_left_raw = pm.math.sigmoid(rand_a - rand_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
