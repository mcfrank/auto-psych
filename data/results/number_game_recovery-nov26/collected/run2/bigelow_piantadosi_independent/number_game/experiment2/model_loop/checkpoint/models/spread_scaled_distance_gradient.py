"""Spread-scaled numerical distance gradient.

People judge a target by its numerical closeness to the nearest example shown,
with willingness to say yes falling off exponentially (a Shepard generalization
gradient) with the target's distance to that example on the 1-100 number line.
The breadth of the gradient is set by how dispersed the examples are: clustered
examples give a narrow gradient, scattered examples a broad one.
"""
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    t = int(target)
    return {
        "nearest_distance": min(abs(t - x) for x in examples) / 100.0,
        "example_range": (max(examples) - min(examples)) / 100.0,
    }


with pm.Model() as model:
    nearest_distance = pm.Data("nearest_distance", np.zeros(1, dtype="float64"))
    example_range = pm.Data("example_range", np.zeros(1, dtype="float64"))

    # Yes-propensity at an example, and how strongly similarity drives yes.
    intercept = pm.Normal("intercept", mu=-1.0, sigma=1.5)
    gain = pm.HalfNormal("gain", sigma=3.0)
    # Gradient breadth: a baseline width plus growth with the examples' spread.
    base_width = pm.LogNormal("base_width", mu=np.log(0.1), sigma=1.0)
    spread_scale = pm.HalfNormal("spread_scale", sigma=1.0)

    width = base_width + spread_scale * example_range
    similarity = pm.math.exp(-nearest_distance / width)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * similarity))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
