"""Log-magnitude exemplar similarity.

People ignore arithmetic rules and generalize by exemplar similarity on a
compressed (logarithmic) mental number line: P(yes) falls off smoothly with the
target's log-distance to the nearest example, and the falloff is wider when the
examples themselves are spread far apart (larger typical gap between them).
"""
import math

import numpy as np
import pymc as pm


def compute_features(number_set, target):
    xs = sorted(math.log(int(v)) for v in number_set.split(","))
    lt = math.log(int(target))
    d = min(abs(lt - x) for x in xs)
    gap = (xs[-1] - xs[0]) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return {"log_nearest_dist": d, "log_mean_gap": gap}


with pm.Model() as model:
    dist = pm.Data("log_nearest_dist", np.zeros(1, dtype="float64"))
    gap = pm.Data("log_mean_gap", np.zeros(1, dtype="float64"))

    # Yes-propensity for a target at an example, steepness of the similarity
    # gradient, and how much the example spread widens it.
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    steepness = pm.HalfNormal("steepness", sigma=5.0)
    spread_gain = pm.HalfNormal("spread_gain", sigma=2.0)

    width = 1.0 + spread_gain * gap
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept - steepness * dist / width)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
