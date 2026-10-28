"""Range-scaled magnitude gradient.

People do not consider arithmetic rules or properties at all; they judge a
target purely by its closeness on the number line to the nearest example, with
a generalization gradient whose width stretches with the spread of the
examples. A single example still supports a fixed baseline width, so "yes"
falls off smoothly with numerical distance, more slowly when the examples
cover a wide range.
"""
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    examples = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    return {
        "near_dist": min(abs(t - x) for x in examples) / 100.0,
        "ex_range": (max(examples) - min(examples)) / 100.0,
    }


with pm.Model() as model:
    near_dist = pm.Data("near_dist", np.zeros(1, dtype="float64"))
    ex_range = pm.Data("ex_range", np.zeros(1, dtype="float64"))

    # log-odds of "yes" for a target at an example
    intercept = pm.Normal("intercept", mu=1.0, sigma=1.5)
    # baseline gradient width (fraction of 1-100) for a single example
    base_width = pm.LogNormal("base_width", mu=np.log(0.1), sigma=1.0)
    # how much the width grows per unit of example range
    range_gain = pm.LogNormal("range_gain", mu=np.log(0.5), sigma=1.0)

    width = base_width + range_gain * ex_range
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept - near_dist / width)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
