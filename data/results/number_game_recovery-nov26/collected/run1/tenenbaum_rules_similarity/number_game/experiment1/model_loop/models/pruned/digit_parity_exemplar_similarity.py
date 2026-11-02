"""Exemplar (generalized context model) account of number generalization.

People do not infer a hidden rule; they store the shown examples as exemplars
and judge a target by its summed similarity to them. Each number is
represented on three psychological dimensions — magnitude, parity and final
digit — and similarity to an example decays exponentially with magnitude
distance and with each mismatch in parity or final digit. "Yes" is the summed
similarity relative to a background response criterion (Luce choice rule).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4  # maximum number of examples in a set


def prepare_observed(rows):
    n = len(rows)
    dist = np.zeros((n, K))
    par = np.zeros((n, K))
    dig = np.zeros((n, K))
    mask = np.zeros((n, K))
    for i, r in enumerate(rows):
        t = int(r["target"])
        xs = [int(v) for v in str(r["number_set"]).split(",")]
        for j, x in enumerate(xs[:K]):
            dist[i, j] = abs(t - x) / 100.0
            par[i, j] = float(t % 2 != x % 2)
            dig[i, j] = float(t % 10 != x % 10)
            mask[i, j] = 1.0
    rating = np.array([int(float(r["rating"])) for r in rows], dtype="int64")
    return {"ex_dist": dist, "ex_parity_mismatch": par,
            "ex_digit_mismatch": dig, "ex_mask": mask, "rating": rating}


with pm.Model() as model:
    ex_dist = pm.Data("ex_dist", np.zeros((1, K)))
    ex_par = pm.Data("ex_parity_mismatch", np.zeros((1, K)))
    ex_dig = pm.Data("ex_digit_mismatch", np.zeros((1, K)))
    ex_mask = pm.Data("ex_mask", np.ones((1, K)))

    # Sensitivity (attention-weighted specificity) on each dimension.
    c_mag = pm.HalfNormal("c_mag", sigma=20.0)
    c_parity = pm.HalfNormal("c_parity", sigma=3.0)
    c_digit = pm.HalfNormal("c_digit", sigma=3.0)
    # Background response criterion (log scale).
    log_b = pm.Normal("log_b", mu=0.0, sigma=2.0)

    sim = pt.exp(-c_mag * ex_dist - c_parity * ex_par - c_digit * ex_dig) * ex_mask
    summed = pt.sum(sim, axis=1)
    p = summed / (summed + pt.exp(log_b))
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
