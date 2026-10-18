"""Range-extension generalization scaled by example spacing.

People ignore arithmetic rules and generalize by range extension: every number
between the smallest and largest example is treated as equally likely, and
willingness to say yes falls off smoothly with how far a target lies beyond
that range, measured in units of the typical spacing between the examples
(mean gap for 2+ examples; a learned width for a single example).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    """Distance beyond the example range, the mean example gap, and a single-example flag."""
    xs = sorted(int(x) for x in str(number_set).split(","))
    t = int(target)
    lo, hi = xs[0], xs[-1]
    outside = max(lo - t, t - hi, 0)
    single = 1.0 if len(xs) == 1 else 0.0
    gap = 0.0 if len(xs) == 1 else (hi - lo) / (len(xs) - 1)
    return {"range_dist": outside / 10.0, "mean_gap": gap / 10.0, "single_example": single}


with pm.Model() as model:
    range_dist = pm.Data("range_dist", np.zeros(1, dtype="float64"))
    mean_gap = pm.Data("mean_gap", np.zeros(1, dtype="float64"))
    single_example = pm.Data("single_example", np.zeros(1, dtype="float64"))

    # Logit of yes for a target inside the range; how fast yes falls off per
    # spacing unit beyond it; the generalization width around a lone example.
    inside_logit = pm.Normal("inside_logit", mu=0.0, sigma=2.0)
    falloff = pm.HalfNormal("falloff", sigma=2.0)
    single_width = pm.LogNormal("single_width", mu=0.0, sigma=1.0)

    scale = single_example * single_width + (1.0 - single_example) * (mean_gap + 0.1)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(inside_logit - falloff * range_dist / scale))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
