"""Final-digit similarity with span falloff.

A refinement of final_digit_similarity_gradient: "yes" grows with the target's
average final-digit similarity to the examples (same digit = 1, same parity
= partial, opposite parity = 0), and, as the one change, the log-odds of "yes"
fall off linearly with how far the target lies outside the span from the
smallest to the largest example (no penalty inside the span).
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
        "dist_beyond_span": float(beyond),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    beyond = pm.Data("dist_beyond_span", np.zeros(1, dtype="float64"))

    # Similarity of a same-parity, different-digit ending on the final-digit metric.
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # Drop in log-odds per 100 units the target lies beyond the examples' span.
    falloff = pm.HalfNormal("falloff", sigma=3.0)
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + gain * mean_sim - falloff * beyond)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
