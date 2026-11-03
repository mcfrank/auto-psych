"""Independent-feature category prototype.

People summarise the examples as one category prototype: a central tendency and
spread on a compressed (log) number line, plus the frequencies of each final
digit and each parity among the examples (shrunk toward uniform by prior
uncertainty).  They say yes in proportion to how typical the target is of that
prototype relative to a number picked at random, with every feature judged
independently (a single-cluster rational / naive-Bayes category density).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

GRID = np.arange(1, 101)
LOG_GRID = np.log(GRID).astype("float64")
MU0 = float(LOG_GRID.mean())


def prepare_observed(rows):
    sets, set_idx = {}, []
    for r in rows:
        s = r["number_set"]
        if s not in sets:
            sets[s] = len(sets)
        set_idx.append(sets[s])
    n_s = len(sets)
    n_ex = np.zeros(n_s)
    xbar = np.zeros(n_s)
    ss = np.zeros(n_s)
    for s, i in sets.items():
        ex = np.array([int(x) for x in s.split(",")], dtype=float)
        lx = np.log(ex)
        n_ex[i] = len(ex)
        xbar[i] = lx.mean()
        ss[i] = ((lx - lx.mean()) ** 2).sum()
    digit_count = np.zeros(len(rows))
    parity_count = np.zeros(len(rows))
    tgt = np.zeros(len(rows), dtype="int64")
    for j, r in enumerate(rows):
        ex = [int(x) for x in r["number_set"].split(",")]
        t = int(r["target"])
        tgt[j] = t - 1
        digit_count[j] = sum(1 for e in ex if e % 10 == t % 10)
        parity_count[j] = sum(1 for e in ex if e % 2 == t % 2)
    return {
        "set_idx": np.array(set_idx, dtype="int64"),
        "target_idx": tgt,
        "set_n": n_ex,
        "set_xbar": xbar,
        "set_ss": ss,
        "digit_count": digit_count,
        "parity_count": parity_count,
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))
    set_n = pm.Data("set_n", np.ones(1, dtype="float64"))
    set_xbar = pm.Data("set_xbar", np.zeros(1, dtype="float64"))
    set_ss = pm.Data("set_ss", np.zeros(1, dtype="float64"))
    digit_count = pm.Data("digit_count", np.zeros(1, dtype="float64"))
    parity_count = pm.Data("parity_count", np.zeros(1, dtype="float64"))

    # Prior expected spread of a category on the log number line.
    prior_sd = pm.LogNormal("prior_sd", mu=np.log(0.5), sigma=0.7)
    # Prior uncertainty (pseudo-counts) about final-digit and parity frequencies.
    alpha_digit = pm.LogNormal("alpha_digit", mu=0.0, sigma=1.0)
    alpha_parity = pm.LogNormal("alpha_parity", mu=0.0, sigma=1.0)
    # How typicality maps onto the yes response.
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=1.5)

    kappa0, nu0 = 1.0, 2.0
    kappa_n = kappa0 + set_n
    mu_n = (kappa0 * MU0 + set_n * set_xbar) / kappa_n
    nu_n = nu0 + set_n
    s2_n = (nu0 * prior_sd**2 + set_ss
            + kappa0 * set_n * (set_xbar - MU0) ** 2 / kappa_n) / nu_n
    scale2 = s2_n * (1.0 + 1.0 / kappa_n)
    # Student-t predictive over the 100 numbers, normalised on the discrete grid.
    z2 = (LOG_GRID[None, :] - mu_n[:, None]) ** 2 / scale2[:, None]
    logf = -0.5 * (nu_n[:, None] + 1.0) * pt.log1p(z2 / nu_n[:, None])
    log_mag = logf - pt.logsumexp(logf, axis=1, keepdims=True) + np.log(100.0)
    mag_term = log_mag[set_idx, target_idx]

    n_t = set_n[set_idx]
    digit_term = pt.log((digit_count + alpha_digit) / (n_t + 10.0 * alpha_digit)) + np.log(10.0)
    parity_term = pt.log((parity_count + alpha_parity) / (n_t + 2.0 * alpha_parity)) + np.log(2.0)

    typicality = mag_term + digit_term + parity_term
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(intercept + slope * typicality), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
