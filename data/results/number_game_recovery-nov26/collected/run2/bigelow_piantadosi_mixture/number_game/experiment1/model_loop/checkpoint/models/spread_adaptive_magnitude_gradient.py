"""Spread-adaptive magnitude similarity.

People judge a target by its numerical closeness to the nearest shown example,
with an exponentially decaying (Shepard) generalization gradient whose width
scales with how spread out the examples are: clustered examples give a sharp
gradient, scattered examples a broad one. No rules or digit features.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    return {
        "nearest_dist": min(abs(t - x) for x in xs) / 100.0,
        "example_spread": (max(xs) - min(xs)) / 100.0,
    }


with pm.Model() as model:
    nearest_dist = pm.Data("nearest_dist", np.zeros(1, dtype="float64"))
    example_spread = pm.Data("example_spread", np.zeros(1, dtype="float64"))

    # Yes-rate far from every example (floor) and at an example (ceiling).
    floor = pm.Beta("floor", alpha=2.0, beta=8.0)
    peak_gain = pm.Beta("peak_gain", alpha=4.0, beta=2.0)
    ceiling = floor + (1.0 - floor) * peak_gain

    # Gradient width (fraction of the 1-100 range): base width plus a part
    # proportional to the spread of the examples.
    base_width = pm.HalfNormal("base_width", sigma=0.1)
    spread_gain = pm.HalfNormal("spread_gain", sigma=1.0)
    width = base_width + spread_gain * example_spread + 1e-3

    sim = pt.exp(-nearest_dist / width)
    p = floor + (ceiling - floor) * sim
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
