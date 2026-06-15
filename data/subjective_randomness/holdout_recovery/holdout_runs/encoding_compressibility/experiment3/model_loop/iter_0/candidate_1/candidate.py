import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Deviation-from-expectation model: both too-few and too-many alternations look
# non-random, so |p_alts - 0.5| is a symmetric non-randomness cue. This differs
# from all existing baselines, which treat p_alts linearly (or ignore it).

with pm.Model() as model:
    max_run_norm_a = pm.Data("max_run_norm_a", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))

    max_run_norm_b = pm.Data("max_run_norm_b", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Weights constrained positive: higher violation → looks less random.
    w_run = pm.HalfNormal("w_run", sigma=5.0)
    w_alts_dev = pm.HalfNormal("w_alts_dev", sigma=5.0)
    w_imbalance = pm.HalfNormal("w_imbalance", sigma=5.0)

    tau = pm.HalfNormal("tau", sigma=2.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # Non-randomness score: |p_alts - 0.5| measures deviation from the iid
    # fair-coin alternation expectation (symmetric: penalises both runs and
    # perfect alternation).
    violation_a = (
        w_run * max_run_norm_a
        + w_alts_dev * pt.abs(p_alts_a - 0.5)
        + w_imbalance * imbalance_a
    )
    violation_b = (
        w_run * max_run_norm_b
        + w_alts_dev * pt.abs(p_alts_b - 0.5)
        + w_imbalance * imbalance_b
    )

    # Left chosen when it has the lower violation score (appears more random).
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(tau * (violation_b - violation_a) + side_bias),
    )

    pm.Bernoulli("response", p=p_left, observed=chose_left)
