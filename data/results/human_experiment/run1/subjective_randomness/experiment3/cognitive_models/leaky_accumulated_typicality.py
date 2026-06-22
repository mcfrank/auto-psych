"""
People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its events, but working memory for past events is leaky. The evidence contributed by each event decays exponentially over time, meaning the total accumulated randomness score saturates for longer sequences, preventing large differences in sequence length from having an oversized effect on choice.
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

    # Retention rate for leaky working memory
    gamma_raw = pm.Beta("gamma", alpha=2.0, beta=2.0)
    gamma = pt.clip(gamma_raw, 1e-6, 1.0 - 1e-6)

    # Calculate empirical rates
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

    # Effective length under exponential decay: sum_{i=0}^{n-1} gamma^i = (1 - gamma^n) / (1 - gamma)
    eff_len_a = (1.0 - pt.pow(gamma, pt.cast(n_a, "float64"))) / (1.0 - gamma)
    eff_len_b = (1.0 - pt.pow(gamma, pt.cast(n_b, "float64"))) / (1.0 - gamma)

    # Total randomness score uses effective length
    rand_a = eff_len_a * typ_a
    rand_b = eff_len_b * typ_b

    # Choice probability
    p_left_raw = pm.math.sigmoid(rand_a - rand_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
