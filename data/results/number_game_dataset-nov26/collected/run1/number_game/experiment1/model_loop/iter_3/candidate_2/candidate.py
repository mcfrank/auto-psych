"""Feature-distribution prototype (naive-Bayes category estimation).

People summarise the examples into a category description: a normal
distribution of magnitudes (mean of the examples, spread = their variance plus
prior pseudo-variance, widened for predictive uncertainty) and hierarchically
smoothed frequencies of final digits (smoothed toward parity frequencies, which
are themselves smoothed toward chance). They say yes according to the log
likelihood ratio of the target under this category versus a uniformly random
number in 1-100.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = len(ex)
    m = float(np.mean(ex))
    var = float(np.var(ex)) if n > 1 else 0.0
    return {
        "fd_n": float(n),
        "fd_sqdev": ((t - m) / 100.0) ** 2,
        "fd_var": var / 10000.0,
        "fd_digit_count": float(sum(1 for x in ex if x % 10 == t % 10)),
        "fd_parity_count": float(sum(1 for x in ex if x % 2 == t % 2)),
    }


with pm.Model() as model:
    n = pm.Data("fd_n", np.ones(1, dtype="float64"))
    sqdev = pm.Data("fd_sqdev", np.zeros(1, dtype="float64"))
    var = pm.Data("fd_var", np.zeros(1, dtype="float64"))
    dcount = pm.Data("fd_digit_count", np.zeros(1, dtype="float64"))
    pcount = pm.Data("fd_parity_count", np.zeros(1, dtype="float64"))

    # prior magnitude spread (on 0-1 scale), digit and parity smoothing strengths
    tau = pm.HalfNormal("tau", sigma=0.3)
    alpha_digit = pm.HalfNormal("alpha_digit", sigma=3.0)
    alpha_parity = pm.HalfNormal("alpha_parity", sigma=3.0)
    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=1.0)

    s2 = (var + tau ** 2 + 1e-4) * (1.0 + 1.0 / n)
    # log normal density on 0-1 scale vs uniform density 1
    log_lr_mag = -0.5 * pt.log(2 * np.pi * s2) - 0.5 * sqdev / s2

    p_par = (pcount + alpha_parity * 0.5) / (n + alpha_parity + 1e-6)
    p_dig = (dcount + (alpha_digit + 1e-6) * p_par / 5.0) / (n + alpha_digit + 1e-6)
    log_lr_dig = pt.log(p_dig + 1e-9) - np.log(0.1)

    score = bias + gain * (log_lr_mag + log_lr_dig)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(score), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
