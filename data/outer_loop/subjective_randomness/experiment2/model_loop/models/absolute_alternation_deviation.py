"""People judge a sequence as less random the further its number of alternations deviates from the expected number under their subjective ideal rate, computing distance in absolute counts rather than proportions."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))

    ideal_rate = pm.Beta("ideal_rate", alpha=2.0, beta=2.0)
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Expected number of alternations under ideal rate
    expected_alts_a = pt.cast(n_a - 1, "float64") * ideal_rate
    expected_alts_b = pt.cast(n_b - 1, "float64") * ideal_rate

    # Absolute deviation in counts
    dist_a = pt.abs(pt.cast(alts_a, "float64") - expected_alts_a)
    dist_b = pt.abs(pt.cast(alts_b, "float64") - expected_alts_b)

    # Prefer sequence with smaller absolute deviation
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (dist_b - dist_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
