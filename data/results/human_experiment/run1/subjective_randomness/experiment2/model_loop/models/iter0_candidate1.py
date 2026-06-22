"""
People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its events, but working memory for past events is leaky. The evidence contributed by each event decays exponentially over time, meaning the total accumulated randomness score saturates for longer sequences, which prevents large differences in sequence length from having an oversized effect on choice.
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
    
    # Memory decay parameter for the leaky accumulator
    # A decay of 1 means perfect memory (linear accumulation), 0 means no memory (only last event matters)
    decay = pm.Beta("decay", alpha=2.0, beta=2.0)

    # Calculate empirical rates (safeguard against division by zero)
    p_a = h_a / pt.maximum(n_a, 1)
    p_b = h_b / pt.maximum(n_b, 1)

    alt_rate_a = alts_a / pt.maximum(n_a - 1, 1)
    alt_rate_b = alts_b / pt.maximum(n_b - 1, 1)

    # Calculate average per-event typicality
    typ_a = base_typ - (
        w_p * pt.square(p_a - ideal_p) + w_alt * pt.square(alt_rate_a - ideal_alt)
    )
    typ_b = base_typ - (
        w_p * pt.square(p_b - ideal_p) + w_alt * pt.square(alt_rate_b - ideal_alt)
    )

    # Calculate effective length under exponential decay
    # Sum of geometric series: 1 + decay + decay^2 + ... + decay^(n-1) = (1 - decay^n) / (1 - decay)
    # We clip decay to 0.999 to avoid division by zero
    safe_decay = pt.clip(decay, 1e-6, 0.999)
    eff_len_a = (1.0 - pt.power(safe_decay, pt.cast(n_a, "float64"))) / (1.0 - safe_decay)
    eff_len_b = (1.0 - pt.power(safe_decay, pt.cast(n_b, "float64"))) / (1.0 - safe_decay)

    # Total randomness score with leaky accumulation
    rand_a = eff_len_a * typ_a
    rand_b = eff_len_b * typ_b

    # Choice probability
    p_left_raw = pm.math.sigmoid(rand_a - rand_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
