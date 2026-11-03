"""People judge a target by its similarity to the nearest example on a
logarithmically compressed mental number line: yes-probability falls off
smoothly with the log-ratio distance between the target and the closest
example, so generalisation is wide around large examples and tight around
small ones, with no rule inference."""
import math

import numpy as np
import pymc as pm


def compute_features(number_set, target):
    examples = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    d = min(abs(math.log(t) - math.log(x)) for x in examples)
    return {"log_ratio_distance": float(d)}


with pm.Model() as model:
    log_ratio_distance = pm.Data("log_ratio_distance", np.zeros(1, dtype="float64"))

    # Yes-propensity for a target at an example, and the steepness of the
    # similarity gradient per unit of log-ratio distance.
    intercept = pm.Normal("intercept", mu=1.0, sigma=2.0)
    falloff = pm.HalfNormal("falloff", sigma=10.0)

    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept - falloff * log_ratio_distance)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
