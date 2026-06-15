"""Prototype similarity model scaled by sequence length.

Under a fair coin, the standard deviation of p_alts shrinks as 1/sqrt(n-1) and
the standard deviation of the head proportion shrinks as 1/sqrt(n), so the same
absolute deviation is statistically more surprising — and should feel less random
— in a longer sequence. This model multiplies each penalty by sqrt(n), equivalent
to using a z-score-style evidence weighting."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight
    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")

    score_a = -(
        balance_weight * imbalance_a * pt.sqrt(n_a_f)
        + alt_weight * pt.abs(p_alts_a - theta_alt) * pt.sqrt(pt.maximum(n_a_f - 1.0, 1.0))
    )
    score_b = -(
        balance_weight * imbalance_b * pt.sqrt(n_b_f)
        + alt_weight * pt.abs(p_alts_b - theta_alt) * pt.sqrt(pt.maximum(n_b_f - 1.0, 1.0))
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
