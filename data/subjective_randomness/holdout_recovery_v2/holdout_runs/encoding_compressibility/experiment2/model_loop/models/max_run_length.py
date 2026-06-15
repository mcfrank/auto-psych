"""People judge a sequence as more random when its longest consecutive run is shorter:
a long run (e.g., HHHHH) is the most compact single-symbol run-length encoding unit,
and its presence is the single most salient cue that the sequence is structured rather
than random."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # Shorter max run → less compressible → more random-looking
    score_a = -max_run_norm_a
    score_b = -max_run_norm_b

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
