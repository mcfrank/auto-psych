"""Spacing-scaled numerical distance gradient.

People judge a target by its numerical distance to the nearest example: the
probability of saying yes falls off exponentially (a Shepard generalization
gradient) with that distance. The gradient's width scales with how widely the
examples are spaced (mean gap between sorted examples), so clustered examples
give a narrow gradient and widely spaced examples a broad one.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    xs = sorted(int(x) for x in str(number_set).split(","))
    t = int(target)
    dist = min(abs(t - x) for x in xs)
    gap = (xs[-1] - xs[0]) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return {"nearest_dist": dist / 10.0, "mean_gap": gap / 10.0}


with pm.Model() as model:
    nearest_dist = pm.Data("nearest_dist", np.zeros(1, dtype="float64"))
    mean_gap = pm.Data("mean_gap", np.zeros(1, dtype="float64"))

    # Gradient width (in units of 10): a base width plus a share of the spacing.
    base_width = pm.HalfNormal("base_width", sigma=2.0)
    gap_scale = pm.HalfNormal("gap_scale", sigma=2.0)
    # Response floor (yes to far targets) and ceiling (yes at an example).
    floor = pm.Beta("floor", alpha=2.0, beta=5.0)
    ceiling = pm.Beta("ceiling", alpha=5.0, beta=2.0)

    width = base_width + gap_scale * mean_gap + 0.05
    sim = pt.exp(-nearest_dist / width)
    p = floor * (1.0 - sim) + ceiling * sim
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
