"""Relational step extrapolation.

People do not look for shared properties or similarity; they infer the
generating operation from the relations among the examples themselves. They
notice the differences and whole-number ratios linking pairs of examples
(e.g. 16 is 4 times 4, 83 is 15 below 98) and say "yes" to a target that the
same step produces from one of the examples (64 after 4,16,12; 68 after
98,83), and "no" otherwise; a lone example carries no relation to extrapolate.
"""
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    diffs, ratios = set(), set()
    for a in xs:
        for b in xs:
            if a != b:
                diffs.add(abs(a - b))
                if a % b == 0 and a // b > 1:
                    ratios.add(a // b)
    by_diff = any(abs(t - x) == d for x in xs for d in diffs)
    by_ratio = any(t == x * q or x == t * q for x in xs for q in ratios)
    return {"reach_by_difference": float(by_diff), "reach_by_ratio": float(by_ratio)}


with pm.Model() as model:
    by_diff = pm.Data("reach_by_difference", np.zeros(1, dtype="float64"))
    by_ratio = pm.Data("reach_by_ratio", np.zeros(1, dtype="float64"))

    # Baseline log-odds of yes for a target no inferred step produces, and the
    # boost when the same additive or multiplicative step reaches the target.
    intercept = pm.Normal("intercept", mu=0.0, sigma=1.5)
    gain_diff = pm.HalfNormal("gain_diff", sigma=2.0)
    gain_ratio = pm.HalfNormal("gain_ratio", sigma=2.0)
    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(intercept + gain_diff * by_diff + gain_ratio * by_ratio),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
