"""Exemplar similarity (generalized context model) for the number game.

People do not infer a hidden rule; they judge a target by its summed
similarity to the remembered examples. Similarity to each example decays
exponentially with a city-block psychological distance made of the two
numbers' magnitude difference and whether their tens digits and their ones
digits differ; "yes" grows with the target's total similarity to the examples
relative to a response criterion (Luce choice rule).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4


def _digits(n):
    return n // 10, n % 10


def prepare_observed(rows):
    """Per-trial (trials, 4) distance components to each example, padded and masked."""
    n = len(rows)
    mag = np.zeros((n, MAX_EX))
    tens = np.zeros((n, MAX_EX))
    ones = np.zeros((n, MAX_EX))
    mask = np.zeros((n, MAX_EX))
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")][:MAX_EX]
        t = int(r["target"])
        tt, to = _digits(t)
        for j, x in enumerate(ex):
            xt, xo = _digits(x)
            mag[i, j] = abs(t - x) / 100.0
            tens[i, j] = float(tt != xt)
            ones[i, j] = float(to != xo)
            mask[i, j] = 1.0
    return {
        "mag_dist": mag,
        "tens_mismatch": tens,
        "ones_mismatch": ones,
        "example_mask": mask,
        "rating": np.array([int(r.get("rating", 0)) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    mag_dist = pm.Data("mag_dist", np.zeros((1, MAX_EX)))
    tens_mismatch = pm.Data("tens_mismatch", np.zeros((1, MAX_EX)))
    ones_mismatch = pm.Data("ones_mismatch", np.zeros((1, MAX_EX)))
    example_mask = pm.Data("example_mask", np.ones((1, MAX_EX)))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Attention-weighted sensitivity to each distance component.
    w_mag = pm.HalfNormal("w_mag", sigma=10.0)
    w_tens = pm.HalfNormal("w_tens", sigma=3.0)
    w_ones = pm.HalfNormal("w_ones", sigma=3.0)
    # Log response criterion: similarity needed for an even yes/no split.
    log_criterion = pm.Normal("log_criterion", mu=-1.0, sigma=2.0)

    dist = w_mag * mag_dist + w_tens * tens_mismatch + w_ones * ones_mismatch
    log_sim = pt.where(example_mask > 0, -dist, -1e4)
    n_ex = pt.sum(example_mask, axis=1)
    log_mean_sim = pt.logsumexp(log_sim, axis=1) - pt.log(n_ex)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(log_mean_sim - log_criterion), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)
