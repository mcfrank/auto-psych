"""PyMC adapter for the Hahn & Warren (2009) finite-window model family.

Randomness peaks when the longest run is typical of a fair coin viewed through a
limited memory window: e = log2(min(n, window)). Runs longer than e are penalised
(the streak-aversion effect); runs shorter than e are penalised by the smaller
``over_alt_penalty`` weight. See the pure-Python twin in
``model_families/window_typicality.py`` for the full rationale.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _score(n, max_run, window, over_alt_penalty):
    n_f = pt.cast(n, "float64")
    max_run_f = pt.cast(max_run, "float64")
    expected_run = pt.log2(pt.minimum(n_f, window))
    too_long = pt.softplus(max_run_f - expected_run)
    too_short = pt.softplus(expected_run - max_run_f)
    return -(too_long + over_alt_penalty * too_short)


with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    window = pm.Uniform("window", lower=2.0, upper=8.0)
    over_alt_penalty = pm.Uniform("over_alt_penalty", lower=0.0, upper=1.0)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    score_a = _score(n_a, max_run_a, window, over_alt_penalty)
    score_b = _score(n_b, max_run_b, window, over_alt_penalty)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
