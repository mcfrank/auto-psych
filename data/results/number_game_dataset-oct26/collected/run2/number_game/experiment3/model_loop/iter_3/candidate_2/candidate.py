"""Single-retrieval probability matching.

People decide by probability matching on a single retrieved memory rather
than by thresholding summed familiarity: on each question they recall one of
the examples (the more recently shown, the likelier it comes to mind), accept
the target with a probability equal to how similar it is to that example (in
size, final digit, parity and shared factor), and if that example does not
vouch for the target they fall back on a fixed guessing rate of "yes". The
yes-rate thus rises linearly with similarity from a guessing floor to a
ceiling below certainty, with no sigmoid sharpening of the evidence.
"""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def _factor_mismatch(t, x):
    m = min(t, x)
    if m <= 1:
        return 0.0
    return 1.0 - math.log(math.gcd(t, x)) / math.log(m)


def prepare_observed(rows):
    n = len(rows)
    mism = np.ones((n, 4, MAXK))
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")][:MAXK]
        t = int(row["target"])
        k = len(ex)
        for i, x in enumerate(ex):
            mism[r, 0, i] = abs(t - x) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        rv = row.get("rating")
        rating[r] = int(float(rv)) if rv is not None and str(rv).strip() != "" else 0
    return {"mismatch": mism, "lag": lag, "ex_mask": mask, "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.ones((1, 4, MAXK)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))

    # Attention to size (per 100 units), final digit, parity and shared factor.
    w = pm.HalfNormal("w", sigma=np.array([10.0, 2.0, 2.0, 2.0]), shape=4)
    # How much likelier the next-more-recent example is to be retrieved.
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    # Probability a perfectly matching retrieved example vouches for the target.
    accept_max = pm.Beta("accept_max", alpha=4.0, beta=1.5)
    # Guessing rate of "yes" when the retrieved example does not vouch.
    guess_yes = pm.Beta("guess_yes", alpha=1.5, beta=3.0)

    sim = pt.exp(-pt.sum(w[None, :, None] * mismatch, axis=1))
    retr = ex_mask * pt.exp(lag * pt.log(retention))
    retr = retr / pt.sum(retr, axis=1, keepdims=True)
    accept = accept_max * pt.sum(retr * sim, axis=1)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(accept + (1.0 - accept) * guess_yes, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
