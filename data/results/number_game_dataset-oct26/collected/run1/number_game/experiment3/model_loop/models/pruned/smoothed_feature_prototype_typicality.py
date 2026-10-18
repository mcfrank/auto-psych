"""Smoothed feature-frequency prototype typicality.

People do not infer a rule; they abstract a prototype of the examples as a
profile of how often each number feature occurs among them (each final digit,
odd/even, each decade, and divisibility by 3, 4 and 5), blending these observed
frequencies with a weak prior expectation that any feature value is possible,
and they say "yes" according to how typical the target is under that profile.
As examples accumulate the prior is diluted, so a feature value seen in every
example becomes ever more expected and an unseen one ever more atypical.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    out = {
        "n_examples": float(len(xs)),
        "cnt_digit": float(sum(x % 10 == t % 10 for x in xs)),
        "cnt_parity": float(sum(x % 2 == t % 2 for x in xs)),
        "cnt_decade": float(sum(x // 10 == t // 10 for x in xs)),
    }
    for k in (3, 4, 5):
        out[f"cnt_div{k}"] = float(sum((x % k == 0) == (t % k == 0) for x in xs))
    return out


def _log_typicality(count, n, alpha, k):
    # log of (smoothed profile probability / chance probability 1/k)
    return pt.log(k * (count + alpha / k) / (n + alpha))


with pm.Model() as model:
    n = pm.Data("n_examples", np.ones(1, dtype="float64"))
    c_digit = pm.Data("cnt_digit", np.zeros(1, dtype="float64"))
    c_par = pm.Data("cnt_parity", np.zeros(1, dtype="float64"))
    c_dec = pm.Data("cnt_decade", np.zeros(1, dtype="float64"))
    c_d3 = pm.Data("cnt_div3", np.zeros(1, dtype="float64"))
    c_d4 = pm.Data("cnt_div4", np.zeros(1, dtype="float64"))
    c_d5 = pm.Data("cnt_div5", np.zeros(1, dtype="float64"))

    # Prior pseudo-count: how strongly people expect unseen feature values.
    alpha = pm.LogNormal("alpha", mu=0.0, sigma=1.0)
    w_digit = pm.HalfNormal("w_digit", sigma=2.0)
    w_parity = pm.HalfNormal("w_parity", sigma=2.0)
    w_decade = pm.HalfNormal("w_decade", sigma=2.0)
    w_div = pm.HalfNormal("w_div", sigma=2.0)
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)

    typicality = (
        w_digit * _log_typicality(c_digit, n, alpha, 10.0)
        + w_parity * _log_typicality(c_par, n, alpha, 2.0)
        + w_decade * _log_typicality(c_dec, n, alpha, 10.0)
        + w_div * (
            _log_typicality(c_d3, n, alpha, 2.0)
            + _log_typicality(c_d4, n, alpha, 2.0)
            + _log_typicality(c_d5, n, alpha, 2.0)
        )
    )
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + typicality))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
