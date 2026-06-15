import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Prototype alternation rate (free; people expect slightly above 0.5).
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    # Relative importance of three randomness cues.
    weights = pm.Dirichlet("weights", a=np.ones(3))
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = -(
        weights[0] * pt.abs(p_alts_a - theta_alt)
        + weights[1] * imbalance_a
        + weights[2] * max_run_norm_a
    )
    score_b = -(
        weights[0] * pt.abs(p_alts_b - theta_alt)
        + weights[1] * imbalance_b
        + weights[2] * max_run_norm_b
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
