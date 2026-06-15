import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Symmetric prior over 4 compressibility-feature weights.
    weights = pm.Dirichlet("weights", a=np.ones(4))
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Alternation deviation: both too-streaky and too-alternating are non-random.
    # |p_alts - 0.5| * 2 maps [0, 0.5, 1] -> [1, 0, 1].
    alt_dev_a = pt.abs(p_alts_a - 0.5) * 2.0
    alt_dev_b = pt.abs(p_alts_b - 0.5) * 2.0

    penalty_a = (
        weights[0] * max_run_norm_a
        + weights[1] * periodicity_a
        + weights[2] * imbalance_a
        + weights[3] * alt_dev_a
    )
    penalty_b = (
        weights[0] * max_run_norm_b
        + weights[1] * periodicity_b
        + weights[2] * imbalance_b
        + weights[3] * alt_dev_b
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (penalty_b - penalty_a) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
