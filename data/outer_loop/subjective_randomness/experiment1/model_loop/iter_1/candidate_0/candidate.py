import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs.
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    max_run_a = pm.Data("max_run_a", np.zeros(1, dtype="int64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    max_run_b = pm.Data("max_run_b", np.zeros(1, dtype="int64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))

    # Sensitivity and mixing weight between the two randomness cues.
    tau = pm.HalfNormal("tau", sigma=2.0)
    w = pm.Beta(
        "w", alpha=1.0, beta=1.0
    )  # w=1 → pure normalized-run; w=0 → pure alternation

    # Cue 1: shorter length-normalized max run → more random-looking.
    norm_run_a = pt.cast(max_run_a, "float64") / pt.cast(n_a, "float64")
    norm_run_b = pt.cast(max_run_b, "float64") / pt.cast(n_b, "float64")
    run_diff = norm_run_b - norm_run_a

    # Cue 2: higher alternation rate → more random-looking.
    alts_diff = p_alts_a - p_alts_b

    util = tau * (w * run_diff + (1.0 - w) * alts_diff)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(util))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
