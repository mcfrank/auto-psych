"""
People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, but rather than penalizing deviations from an "ideal" alternation rate, they treat alternations as inherently random and thus monotonically rewarding. Because their per-event typicality includes a linear reward for alternation rate rather than a quadratic penalty, sequences with more total alternations naturally accumulate higher randomness scores, explaining both why people prefer highly over-alternating sequences and why they still prefer longer sequences even when both options are heavily clustered.
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
    w_p = pm.HalfNormal("w_p", sigma=5.0)
    w_alt = pm.HalfNormal("w_alt", sigma=5.0)
    base_typ = pm.Normal("base_typ", mu=0.0, sigma=5.0)

    # Calculate empirical rates (safeguard against division by zero)
    p_a = h_a / pt.maximum(n_a, 1)
    p_b = h_b / pt.maximum(n_b, 1)

    alt_rate_a = alts_a / pt.maximum(n_a - 1, 1)
    alt_rate_b = alts_b / pt.maximum(n_b - 1, 1)

    # Per-event typicality: base rate + linear reward for alternations - penalty for head proportion
    typ_a = base_typ + w_alt * alt_rate_a - w_p * pt.square(p_a - ideal_p)
    typ_b = base_typ + w_alt * alt_rate_b - w_p * pt.square(p_b - ideal_p)

    # Total randomness score is accumulated typicality over sequence length
    rand_a = n_a * typ_a
    rand_b = n_b * typ_b

    # Choice probability
    p_left_raw = pm.math.sigmoid(rand_a - rand_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
