"""Spread-scaled exemplar similarity.

People do not infer any mathematical rule; they generalize by numerical
similarity alone, judging a target more likely to be produced the closer it
lies to the individual examples on the number line. The breadth of that
similarity gradient is set by how spread out the examples are: tightly
clustered examples yield a narrow gradient, widely scattered examples a broad
one.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4


def compute_features(number_set, target):
    """Distances (scaled to 0-1) from the target to each example, a mask of
    which example slots are used, and the examples' range (0-1)."""
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    feats = {}
    for i in range(MAX_EX):
        if i < len(ex):
            feats[f"dist_{i}"] = abs(t - ex[i]) / 100.0
            feats[f"used_{i}"] = 1.0
        else:
            feats[f"dist_{i}"] = 1.0
            feats[f"used_{i}"] = 0.0
    feats["ex_range"] = (max(ex) - min(ex)) / 100.0
    return feats


with pm.Model() as model:
    dists = [pm.Data(f"dist_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)]
    used = [pm.Data(f"used_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)]
    ex_range = pm.Data("ex_range", np.zeros(1, dtype="float64"))

    # Baseline gradient width (for a single example / zero spread) and how
    # much the width grows with the examples' spread.
    base_width = pm.LogNormal("base_width", mu=np.log(0.08), sigma=0.8)
    spread_gain = pm.HalfNormal("spread_gain", sigma=1.0)
    # Mapping from similarity to a yes response.
    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    gain = pm.HalfNormal("gain", sigma=5.0)

    width = base_width + spread_gain * ex_range
    n_used = sum(used)
    sim = sum(u * pt.exp(-d / width) for d, u in zip(dists, used)) / n_used

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
