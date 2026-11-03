"""Spread-scaled range generalization.

People do not search over mathematical rules; they treat the examples as a
sample from a band of magnitudes and generalize by how far the target falls
outside the span the examples cover, with a tolerance that grows with how
spread out the examples are. A target inside the span is endorsed at a constant
rate, and endorsement decays exponentially with the distance beyond the nearest
edge, on a length scale tol0 + tol1 * (span of the examples).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    xs = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    lo, hi = min(xs), max(xs)
    outside = max(lo - t, t - hi, 0)
    return {"outside_dist": outside / 100.0, "example_span": (hi - lo) / 100.0}


with pm.Model() as model:
    outside_dist = pm.Data("outside_dist", np.zeros(1, dtype="float64"))
    example_span = pm.Data("example_span", np.zeros(1, dtype="float64"))

    # Endorsement rate for targets within the examples' span.
    inside_logit = pm.Normal("inside_logit", mu=0.0, sigma=2.0)
    # Tolerance beyond the span: a base width plus a part proportional to the span.
    tol0 = pm.HalfNormal("tol0", sigma=0.3)
    tol1 = pm.HalfNormal("tol1", sigma=1.0)
    floor = pm.Beta("floor", 1, 9)

    scale = tol0 + tol1 * example_span + 1e-3
    gen = pm.math.sigmoid(inside_logit) * pt.exp(-outside_dist / scale)
    p_yes = pm.Deterministic("p_yes", pt.clip(floor + (1 - floor) * gen, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
