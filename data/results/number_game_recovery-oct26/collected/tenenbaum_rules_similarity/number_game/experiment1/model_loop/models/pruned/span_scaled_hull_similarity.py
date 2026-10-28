"""Span-scaled hull similarity.

People judge a target by its numerical closeness to the range the examples
span: every target inside the examples' range counts as fully similar, and
similarity outside it decays exponentially with the distance from the nearest
edge of that range. The width of this generalization gradient scales with how
widely the examples are spread, so tightly clustered examples give a steep
fall-off at the range edge and widely spread examples a gentle one; the more
similar the target, the more likely a "yes".
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    """Distance from the target to the examples' range, and the range's width (both /100)."""
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    lo, hi = min(xs), max(xs)
    d = 0.0 if lo <= t <= hi else float(min(abs(t - lo), abs(t - hi)))
    return {"hull_dist": d / 100.0, "hull_span": (hi - lo) / 100.0}


with pm.Model() as model:
    hull_dist = pm.Data("hull_dist", np.zeros(1, dtype="float64"))
    hull_span = pm.Data("hull_span", np.zeros(1, dtype="float64"))

    # Ceiling: probability of "yes" for a fully similar target (inside the range).
    p_max = pm.Beta("p_max", alpha=2.0, beta=2.0)
    # Floor: "yes" rate for a maximally dissimilar target.
    p_min = pm.Beta("p_min", alpha=1.0, beta=5.0)
    # Gradient width = base width + gain * span of examples.
    base_width = pm.HalfNormal("base_width", sigma=0.2)
    width_gain = pm.HalfNormal("width_gain", sigma=1.0)

    width = base_width + width_gain * hull_span + 1e-3
    similarity = pt.exp(-hull_dist / width)
    p = p_min + (p_max - p_min) * similarity
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
