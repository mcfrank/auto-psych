"""Sequential delta-rule property learning.

People build their guess about the program's rule incrementally, in the order
the examples are shown: the first example sets the belief strength of each
arithmetic property (parity, divisibility by 3/4/5/10, final digit, prime,
square, power of 2) it has, and each later example revises every property's
strength by an error-correcting (delta-rule) update at an inferred learning
rate. A slow learner stays anchored on the first example (primacy); a fast one
is dominated by the last (recency). "Yes" grows with the share of the final
belief strength carried by properties the target has, regardless of magnitude.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_PRIMES = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}
_POW2 = {2 ** k for k in range(0, 7)}
_MAXN = 4


def _props(n):
    v = [n % 2 == 0, n % 2 == 1, n % 3 == 0, n % 4 == 0, n % 5 == 0, n % 10 == 0,
         n in _PRIMES, n in _SQUARES, n in _POW2]
    v += [n % 10 == d for d in range(10)]
    return np.array(v, dtype="float64")


def prepare_observed(rows):
    """Per-trial example property matrix (padded to 4 positions), target properties,
    and each position's distance from the last example (for the delta-rule weights)."""
    n_t = len(rows)
    nf = len(_props(1))
    ex_props = np.zeros((n_t, _MAXN, nf))
    tgt_props = np.zeros((n_t, nf))
    lag = np.zeros((n_t, _MAXN))
    is_first = np.zeros((n_t, _MAXN))
    valid = np.zeros((n_t, _MAXN))
    rating = np.zeros(n_t, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")]
        n = len(ex)
        for i, e in enumerate(ex):
            ex_props[r, i] = _props(e)
            lag[r, i] = n - 1 - i
            valid[r, i] = 1.0
        is_first[r, 0] = 1.0
        tgt_props[r] = _props(int(row["target"]))
        rating[r] = int(float(row["rating"]))
    return {"ex_props": ex_props, "tgt_props": tgt_props, "lag": lag,
            "is_first": is_first, "valid": valid, "rating": rating}


with pm.Model() as model:
    ex_props = pm.Data("ex_props", np.zeros((1, _MAXN, 19)))
    tgt_props = pm.Data("tgt_props", np.zeros((1, 19)))
    lag = pm.Data("lag", np.zeros((1, _MAXN)))
    is_first = pm.Data("is_first", np.zeros((1, _MAXN)))
    valid = pm.Data("valid", np.zeros((1, _MAXN)))

    # Learning rate of the delta-rule update, baseline yes-bias, and how sharply
    # the target's property match drives the answer.
    alpha = pm.Beta("alpha", alpha=2.0, beta=2.0)
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    sharpness = pm.HalfNormal("sharpness", sigma=10.0)

    # Delta rule s <- s + alpha (x_i - s), starting from s = x_1: example i's
    # final weight is alpha (1-alpha)^lag, the first example's is (1-alpha)^lag.
    decay = (1.0 - alpha) ** lag
    w = valid * pt.switch(pt.eq(is_first, 1.0), decay, alpha * decay)
    strength = pt.sum(w[:, :, None] * ex_props, axis=1)  # (trials, props)
    match = pt.sum(strength * tgt_props, axis=1) / (pt.sum(strength, axis=1) + 1e-9)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(intercept + sharpness * (match - 0.5)), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
