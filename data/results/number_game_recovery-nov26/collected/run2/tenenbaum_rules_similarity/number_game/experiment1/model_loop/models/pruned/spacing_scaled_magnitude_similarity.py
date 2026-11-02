"""Spacing-scaled magnitude similarity.

People judge a target by its numerical closeness to the nearest example on the
1-100 number line, with no rule inference: the chance of "yes" falls off
smoothly with the distance from the target to the closest example. The width
of this generalization gradient is set by how spread out the examples are (the
mean gap between neighbouring examples, plus a baseline width for single
examples), so widely spaced examples license "yes" to far-away targets while
tightly clustered examples restrict "yes" to targets right beside them.
"""
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    """Nearest-example distance and mean neighbouring-example gap (both /100)."""
    xs = sorted(int(x) for x in str(number_set).split(","))
    t = int(target)
    dist = min(abs(t - x) for x in xs) / 100.0
    spacing = (xs[-1] - xs[0]) / (len(xs) - 1) / 100.0 if len(xs) > 1 else 0.0
    return {"nearest_dist": float(dist), "example_spacing": float(spacing)}


with pm.Model() as model:
    nearest_dist = pm.Data("nearest_dist", np.zeros(1, dtype="float64"))
    example_spacing = pm.Data("example_spacing", np.zeros(1, dtype="float64"))

    # Log-odds of "yes" for a target at an example.
    intercept = pm.Normal("intercept", mu=1.0, sigma=2.0)
    # Baseline gradient width (fraction of the 1-100 scale) added to spacing.
    base_width = pm.HalfNormal("base_width", sigma=0.2)
    # How many units of log-odds are lost per gradient width of distance.
    falloff = pm.HalfNormal("falloff", sigma=5.0)

    width = base_width + example_spacing + 1e-3
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept - falloff * nearest_dist / width)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
