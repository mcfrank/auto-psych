"""Log-ratio similarity gradient with spread-adaptive tolerance.

People judge a target by its similarity in magnitude to the examples, measured
as the ratio distance (difference in log magnitude) to the nearest example.
Willingness to say "yes" falls off smoothly with that distance, and the
tolerance of the gradient widens in proportion to how spread out (in log
magnitude) the examples are.
"""
import math

import numpy as np
import pymc as pm


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    t = math.log(int(target))
    logs = [math.log(x) for x in examples]
    return {
        "log_ratio_dist": min(abs(t - x) for x in logs),
        "log_spread": max(logs) - min(logs),
    }


with pm.Model() as model:
    log_ratio_dist = pm.Data("log_ratio_dist", np.zeros(1, dtype="float64"))
    log_spread = pm.Data("log_spread", np.zeros(1, dtype="float64"))

    # Log-odds of "yes" for a target at an example.
    intercept = pm.Normal("intercept", mu=1.0, sigma=2.0)
    # Baseline tolerance (log units) when all examples coincide.
    tol_base = pm.LogNormal("tol_base", mu=np.log(0.05), sigma=1.0)
    # How much tolerance grows per unit of example log-spread.
    tol_spread = pm.HalfNormal("tol_spread", sigma=0.5)

    tolerance = tol_base + tol_spread * log_spread
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept - log_ratio_dist / tolerance)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
