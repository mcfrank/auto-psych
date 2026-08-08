"""
People evaluate the randomness of a sequence by accumulating a subjective sense of typicality over its length, computing each event's typicality as a penalty based on its distance from a mental prototype. Rather than assuming a strictly linear (City Block) or quadratic (Euclidean) penalty, this distance is computed using a freely inferred exponent (a Minkowski-like metric), allowing the model to naturally capture how human observers disproportionately scale the punishment for extreme feature deviations.
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

    # Exponent for the penalty function (e.g. 1.0 = linear, 2.0 = quadratic)
    # Using a LogNormal prior centered around 0 (so exponent is around 1.0), 
    # but allowing it to flexibly fit the Minkowski distance exponent.
    penalty_power = pm.LogNormal("penalty_power", mu=0.0, sigma=0.5)

    # Calculate empirical rates (safeguard against division by zero)
    n_a_f = pt.cast(pt.maximum(n_a, 1), "float64")
    n_b_f = pt.cast(pt.maximum(n_b, 1), "float64")

    p_a = pt.cast(h_a, "float64") / n_a_f
    p_b = pt.cast(h_b, "float64") / n_b_f

    alt_rate_a = pt.cast(alts_a, "float64") / pt.maximum(pt.cast(n_a - 1, "float64"), 1.0)
    alt_rate_b = pt.cast(alts_b, "float64") / pt.maximum(pt.cast(n_b - 1, "float64"), 1.0)

    # Calculate absolute deviations (add epsilon for numerical safety with powers < 1)
    dev_p_a = pt.abs(p_a - ideal_p) + 1e-6
    dev_alt_a = pt.abs(alt_rate_a - ideal_alt) + 1e-6
    
    dev_p_b = pt.abs(p_b - ideal_p) + 1e-6
    dev_alt_b = pt.abs(alt_rate_b - ideal_alt) + 1e-6

    # Calculate per-event typicality using the non-linear penalty function
    typ_a = base_typ - (
        w_p * pt.pow(dev_p_a, penalty_power) + 
        w_alt * pt.pow(dev_alt_a, penalty_power)
    )
    typ_b = base_typ - (
        w_p * pt.pow(dev_p_b, penalty_power) + 
        w_alt * pt.pow(dev_alt_b, penalty_power)
    )

    # Total randomness score is the accumulated typicality over the sequence length
    rand_a = pt.cast(n_a, "float64") * typ_a
    rand_b = pt.cast(n_b, "float64") * typ_b

    # Choice probability
    p_left_raw = pm.math.sigmoid(rand_a - rand_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
