"""
People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but the typicality of each event is strictly positive and bounded. Instead of an unbounded penalty that can drive the per-event score negative, typicality consists of a positive baseline plus an exponentially decaying similarity to a mental prototype. This guarantees that even highly unnatural sequences accumulate positive randomness as they grow, explaining why people still prefer longer sequences when both options are heavily clustered.
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
    # Uninformative priors so over-alternation bias can be naturally inferred
    ideal_p = pm.Beta("ideal_p", alpha=1.0, beta=1.0)
    ideal_alt = pm.Beta("ideal_alt", alpha=1.0, beta=1.0)

    # Sensitivities to deviations
    w_p = pm.HalfNormal("w_p", sigma=5.0)
    w_alt = pm.HalfNormal("w_alt", sigma=5.0)

    # Strictly positive typicality components
    base_typ = pm.HalfNormal("base_typ", sigma=5.0)
    w_typ = pm.HalfNormal("w_typ", sigma=5.0)

    # Calculate empirical rates (safeguard against division by zero)
    p_a = h_a / pt.maximum(n_a, 1)
    p_b = h_b / pt.maximum(n_b, 1)

    alt_rate_a = alts_a / pt.maximum(n_a - 1, 1)
    alt_rate_b = alts_b / pt.maximum(n_b - 1, 1)

    # Exponentially decaying similarity to the mental prototype
    # Bounded between 0 and 1
    sim_a = pt.exp(
        -(w_p * pt.square(p_a - ideal_p) + w_alt * pt.square(alt_rate_a - ideal_alt))
    )
    sim_b = pt.exp(
        -(w_p * pt.square(p_b - ideal_p) + w_alt * pt.square(alt_rate_b - ideal_alt))
    )

    # Per-event typicality is strictly positive
    typ_a = base_typ + w_typ * sim_a
    typ_b = base_typ + w_typ * sim_b

    # Total randomness score is the accumulated typicality over the sequence length
    rand_a = n_a * typ_a
    rand_b = n_b * typ_b

    # Choice probability
    p_left_raw = pm.math.sigmoid(rand_a - rand_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
