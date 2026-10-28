"""Luce ratio decision rule over recency-weighted exemplar similarity.

People judge a target by its recency-weighted similarity to the remembered
examples (magnitude, final digit, parity, divisibility), as in the simple
exemplar model, but turn that familiarity F into an answer by a Luce ratio
rule against an internal criterion familiarity kappa:
P(yes) = F^gamma / (F^gamma + kappa^gamma), gamma being response determinism.
Ratios rather than differences make weakly familiar targets (and weakly
matching sets) near-certain "no"s, giving more extreme set-level yes rates.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def prepare_observed(rows):
    n = len(rows)
    mag = np.zeros((n, MAXK))
    digit = np.ones((n, MAXK))
    parity = np.ones((n, MAXK))
    nodiv = np.ones((n, MAXK))
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")]
        t = int(row["target"])
        k = len(ex)
        for i, x in enumerate(ex[:MAXK]):
            mag[r, i] = abs(t - x) / 100.0
            digit[r, i] = 0.0 if t % 10 == x % 10 else 1.0
            parity[r, i] = 0.0 if t % 2 == x % 2 else 1.0
            nodiv[r, i] = 0.0 if (t % x == 0 or x % t == 0) else 1.0
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        rating[r] = int(float(row.get("rating", 0) or 0))
    return {"mag_dist": mag, "digit_mismatch": digit, "parity_mismatch": parity,
            "div_mismatch": nodiv, "lag": lag, "ex_mask": mask, "rating": rating}


with pm.Model() as model:
    mag_dist = pm.Data("mag_dist", np.zeros((1, MAXK)))
    digit_mismatch = pm.Data("digit_mismatch", np.zeros((1, MAXK)))
    parity_mismatch = pm.Data("parity_mismatch", np.zeros((1, MAXK)))
    div_mismatch = pm.Data("div_mismatch", np.zeros((1, MAXK)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))

    # Attention weights on the dimensions of psychological distance.
    w_mag = pm.HalfNormal("w_mag", sigma=10.0)
    w_digit = pm.HalfNormal("w_digit", sigma=2.0)
    w_parity = pm.HalfNormal("w_parity", sigma=2.0)
    w_div = pm.HalfNormal("w_div", sigma=2.0)
    # Memory retention per step back in the presentation order.
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    # Luce ratio decision rule: log criterion familiarity and determinism.
    log_kappa = pm.Normal("log_kappa", mu=-1.5, sigma=1.5)
    gamma = pm.HalfNormal("gamma", sigma=3.0)

    dist = (w_mag * mag_dist + w_digit * digit_mismatch
            + w_parity * parity_mismatch + w_div * div_mismatch)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    # F^g / (F^g + kappa^g) = sigmoid(g * (log F - log kappa)).
    log_f = pt.log(familiarity + 1e-12)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(gamma * (log_f - log_kappa)), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
