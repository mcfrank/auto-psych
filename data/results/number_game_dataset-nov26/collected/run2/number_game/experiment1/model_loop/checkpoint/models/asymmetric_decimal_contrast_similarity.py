"""Asymmetric decimal contrast similarity.

People judge a target by its asymmetric (Tversky-style) similarity to every
example at once: similarity falls off with numerical distance and with
mismatches in the decimal form (different final digit, and the divisibility
by 2 and by 5 visible in that digit), and the distances to all examples add
(conjunctive: the target must resemble every example). Properties of the
examples missing from the target cost a fraction alpha of a feature's weight;
properties of the target missing from the examples cost 1 - alpha.
"""
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    xs = [int(v) for v in str(number_set).split(",")]
    t = int(target)
    n = float(len(xs))
    f = {
        "mag_dist": sum(abs(t - x) for x in xs) / (100.0 * n),
        "ones_mismatch": sum(1.0 for x in xs if x % 10 != t % 10) / n,
        "even_ex_not_t": sum(1.0 for x in xs if x % 2 == 0 and t % 2 == 1) / n,
        "even_t_not_ex": sum(1.0 for x in xs if t % 2 == 0 and x % 2 == 1) / n,
        "five_ex_not_t": sum(1.0 for x in xs if x % 5 == 0 and t % 5 != 0) / n,
        "five_t_not_ex": sum(1.0 for x in xs if t % 5 == 0 and x % 5 != 0) / n,
    }
    return f


with pm.Model() as model:
    z = lambda name: pm.Data(name, np.zeros(1, dtype="float64"))
    mag_dist = z("mag_dist")
    ones_mismatch = z("ones_mismatch")
    even_ex_not_t = z("even_ex_not_t")
    even_t_not_ex = z("even_t_not_ex")
    five_ex_not_t = z("five_ex_not_t")
    five_t_not_ex = z("five_t_not_ex")

    intercept = pm.Normal("intercept", mu=1.0, sigma=2.0)
    w_mag = pm.HalfNormal("w_mag", sigma=5.0)
    w_ones = pm.HalfNormal("w_ones", sigma=3.0)
    w_even = pm.HalfNormal("w_even", sigma=3.0)
    w_five = pm.HalfNormal("w_five", sigma=3.0)
    alpha = pm.Beta("alpha", alpha=2.0, beta=2.0)

    dist = (
        w_mag * mag_dist
        + w_ones * ones_mismatch
        + 2.0 * w_even * (alpha * even_ex_not_t + (1.0 - alpha) * even_t_not_ex)
        + 2.0 * w_five * (alpha * five_ex_not_t + (1.0 - alpha) * five_t_not_ex)
    )
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept - dist))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
