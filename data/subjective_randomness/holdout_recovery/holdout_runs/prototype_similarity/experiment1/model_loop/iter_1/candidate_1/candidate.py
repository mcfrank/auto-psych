import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # People have an internal prototype for what "random" looks like in two
    # dimensions independently: expected alternation rate and expected max run
    # length.  Both too-short and too-long runs signal non-randomness
    # (bidirectional), which is the key mechanistic difference from existing
    # models that treat max-run as a monotone penalty.
    theta_alt = pm.Uniform("theta_alt", lower=0.35, upper=0.95)
    theta_run = pm.Uniform("theta_run", lower=0.1, upper=0.6)

    # Relative weight on run-length vs alternation-rate deviations.
    w_run = pm.Uniform("w_run", lower=0.0, upper=1.0)

    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    run_dev_a = pt.abs(max_run_norm_a - theta_run)
    run_dev_b = pt.abs(max_run_norm_b - theta_run)
    alt_dev_a = pt.abs(p_alts_a - theta_alt)
    alt_dev_b = pt.abs(p_alts_b - theta_alt)

    score_a = -(w_run * run_dev_a + (1.0 - w_run) * alt_dev_a)
    score_b = -(w_run * run_dev_b + (1.0 - w_run) * alt_dev_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
