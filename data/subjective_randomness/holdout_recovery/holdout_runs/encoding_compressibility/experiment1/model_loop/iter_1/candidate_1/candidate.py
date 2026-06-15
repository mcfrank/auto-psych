import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    longrun_weight = pm.Uniform("longrun_weight", lower=0.01, upper=0.99)
    periodic_share = pm.Uniform("periodic_share", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Calculate final weights ensuring they sum to 1
    remaining = 1.0 - longrun_weight
    periodic_weight = remaining * periodic_share
    imbalance_weight = remaining * (1.0 - periodic_share)

    # The score is negative "compressibility"
    # Sequences that are easily compressed (high run length, highly periodic, highly imbalanced) are deemed less random
    score_a = -(
        longrun_weight * max_run_norm_a
        + periodic_weight * periodicity_a
        + imbalance_weight * imbalance_a
    )
    score_b = -(
        longrun_weight * max_run_norm_b
        + periodic_weight * periodicity_b
        + imbalance_weight * imbalance_b
    )

    # Choice probability mapping
    p_left = pm.Deterministic(
        "p_left", 
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )

    # Likelihood function
    pm.Bernoulli("response", p=p_left, observed=chose_left)
