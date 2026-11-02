"""Spread-scaled numerical distance gradient.

People judge the target purely by its numerical closeness to the nearest
example on the 1-100 number line, with a Shepard-style exponential
generalization gradient whose width is set by how spread out the examples are:
"yes" falls off exponentially with distance from the nearest example, and
widely spaced examples (large mean gap between neighbouring examples) license
a proportionally broader gradient than tightly clustered ones.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    ex = sorted(int(x) for x in number_set.split(","))
    t = int(target)
    d = min(abs(t - x) for x in ex)
    gap = (ex[-1] - ex[0]) / (len(ex) - 1) if len(ex) > 1 else 0.0
    return {"near_dist": d / 100.0, "mean_gap": gap / 100.0}


with pm.Model() as model:
    near_dist = pm.Data("near_dist", np.zeros(1, dtype="float64"))
    mean_gap = pm.Data("mean_gap", np.zeros(1, dtype="float64"))

    # yes-rate right at an example, and baseline yes-rate far from all examples
    peak = pm.Beta("peak", alpha=4.0, beta=2.0)
    floor = pm.Beta("floor", alpha=1.5, beta=6.0)
    # gradient width (on the 0-1 scale): base width plus a share of the mean gap
    base_width = pm.HalfNormal("base_width", sigma=0.1)
    gap_scale = pm.HalfNormal("gap_scale", sigma=1.0)

    width = base_width + gap_scale * mean_gap + 1e-3
    sim = pt.exp(-near_dist / width)
    p = floor + (peak - floor) * sim
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
