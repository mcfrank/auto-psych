"""Final-digit similarity with range extension.

A refinement of final_digit_similarity_gradient: "yes" still grows with the
target's average final-digit similarity to the examples (same digit = 1,
same parity other digit = partial, opposite parity = 0), but people also
treat the examples' span (min to max) as covered by the program, so a target
inside the span gets an inferred boost that fades smoothly with its distance
beyond the span's edges (for one example, distance from it). The one change
is this range-membership component grafted from range_extension_by_spacing.
"""
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = float(len(xs))
    same_digit = sum(1 for x in xs if x % 10 == t % 10) / n
    same_parity_other = sum(1 for x in xs if x % 2 == t % 2 and x % 10 != t % 10) / n
    beyond = max(min(xs) - t, t - max(xs), 0) / 100.0
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "beyond_span": float(beyond),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    beyond_span = pm.Data("beyond_span", np.zeros(1, dtype="float64"))

    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # Boost for lying within the examples' span and how far past its edges it extends.
    range_boost = pm.HalfNormal("range_boost", sigma=2.0)
    range_width = pm.LogNormal("range_width", mu=np.log(0.1), sigma=1.0)
    in_range = pm.math.exp(-beyond_span / range_width)

    p_yes = pm.Deterministic(
        "p_yes",
        pm.math.sigmoid(intercept + gain * mean_sim + range_boost * (in_range - 0.5)),
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
