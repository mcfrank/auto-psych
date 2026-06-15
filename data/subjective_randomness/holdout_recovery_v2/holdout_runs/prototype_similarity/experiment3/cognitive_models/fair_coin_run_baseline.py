"""People judge a sequence as more random when its maximum run length is short relative
to the expected maximum run length for a fair-coin sequence of the same length. The
fair-coin baseline scales as kappa * log2(n), where kappa is a learned parameter
capturing people's implicit sense of what run length is 'normal' for a sequence of
that size."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


_LOG2 = np.log(2.0)


with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Expected max run under a fair coin scales as kappa * log2(n).
    # For a truly fair coin kappa ≈ 1.0; the prior allows learning from data.
    kappa = pm.HalfNormal("kappa", sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=5.0)

    n_a_f = pt.cast(n_a, "float64")
    n_b_f = pt.cast(n_b, "float64")
    max_run_a_f = pt.cast(max_run_a, "float64")
    max_run_b_f = pt.cast(max_run_b, "float64")

    # Excess run: positive means longer-than-expected (non-random), negative means shorter (random)
    excess_a = max_run_a_f - kappa * pt.log(pt.maximum(n_a_f, 2.0)) / _LOG2
    excess_b = max_run_b_f - kappa * pt.log(pt.maximum(n_b_f, 2.0)) / _LOG2

    # Sequence A is chosen as more random when its excess is smaller than B's
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (excess_b - excess_a)),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
