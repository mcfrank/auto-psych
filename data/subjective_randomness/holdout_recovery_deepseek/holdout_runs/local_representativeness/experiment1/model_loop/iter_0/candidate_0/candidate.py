"""
People evaluate the randomness of a sequence by accumulating a subjective sense of
typicality over its length. Each event contributes a baseline typicality discounted by a
penalty based on its distance from a mental prototype, the distance raised to a freely
inferred Minkowski-like exponent so extreme feature deviations are disproportionately
punished. Unlike a strictly length-proportional accumulation, the total impression grows
as the sequence length raised to a second freely inferred exponent, so additional tosses
can either accelerate or saturate the perceived urgency of how much longer a sequence is.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs (names match responses CSV columns)
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))

    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    ideal_p = pm.Beta("ideal_p", alpha=2.0, beta=2.0)
    ideal_alt = pm.Beta("ideal_alt", alpha=2.0, beta=2.0)
    w_p = pm.HalfNormal("w_p", sigma=5.0)
    w_alt = pm.HalfNormal("w_alt", sigma=5.0)
    base_typ = pm.Normal("base_typ", mu=0.0, sigma=5.0)
    penalty_power = pm.LogNormal("penalty_power", mu=0.0, sigma=0.5)
    length_power = pm.LogNormal("length_power", mu=0.0, sigma=0.3)

    # Safe denomial
    n_a_f = pt.cast(pt.maximum(n_a, 1), "float64")
    n_b_f = pt.cast(pt.maximum(n_b, 1), "float64")

    p_a = pt.cast(h_a, "float64") / n_a_f
    p_b = pt.cast(h_b, "float64") / n_b_f
    alt_rate_a = pt.cast(alts_a, "float64") / pt.maximum(pt.cast(n_a - 1, "float64"), 1.0)
    alt_rate_b = pt.cast(alts_b, "float64") / pt.maximum(pt.cast(n_b - 1, "float64"), 1.0)

    dev_p_a = pt.abs(p_a - ideal_p) + 1e-6
    dev_alt_a = pt.abs(alt_rate_a - ideal_alt) + 1e-6
    dev_p_b = pt.abs(p_b - ideal_p) + 1e-6
    dev_alt_b = pt.abs(alt_rate_b - ideal_alt) + 1e-6

    typ_a = base_typ - (
        w_p * pt.pow(dev_p_a, penalty_power)
        + w_alt * pt.pow(dev_alt_a, penalty_power)
    )
    typ_b = base_typ - (
        w_p * pt.pow(dev_p_b, penalty_power)
        + w_alt * pt.pow(dev_alt_b, penalty_power)
    )

    # Accumulated typicality over the sequence length,
    # with a freely inferred exponent on the length itself.
    rand_a = pt.pow(n_a_f, length_power) * typ_a
    rand_b = pt.pow(n_b_f, length_power) * typ_b

    p_left_raw = pm.math.sigmoid(rand_a - rand_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response: the pm.Data tensor passed directly to observed=
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
