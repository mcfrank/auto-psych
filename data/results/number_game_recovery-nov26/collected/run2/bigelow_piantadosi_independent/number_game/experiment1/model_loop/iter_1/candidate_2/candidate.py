"""Interval-only Bayesian generalization (magnitude-based concept learning).

People generalize from the examples purely by numerical magnitude, ignoring
every arithmetic or digit rule: they infer which contiguous range of 1-100 the
program draws from, favouring tighter ranges that still cover all the examples
(the size principle) under an Erlang prior on range width, and say "yes" with
the probability that the target lies inside the inferred range. A small lapse
rate allows random answers.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

SIZES = np.arange(1, 101, dtype="float64")


def _count_intervals(lo, hi):
    """Number of intervals of each size 1..100 within 1..100 covering [lo, hi]."""
    s = SIZES
    upper = np.minimum(lo, 101 - s)
    lower = np.maximum(1, hi - s + 1)
    return np.maximum(0.0, upper - lower + 1)


def prepare_observed(rows):
    n_ex, c_t, c_0, ratings = [], [], [], []
    for r in rows:
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(float(r["target"]))
        m, M = min(ex), max(ex)
        c_0.append(_count_intervals(m, M))
        c_t.append(_count_intervals(min(m, t), max(M, t)))
        n_ex.append(float(len(ex)))
        ratings.append(int(float(r["rating"])))
    return {
        "n_examples": np.asarray(n_ex, dtype="float64"),
        "count_target": np.asarray(c_t, dtype="float64"),
        "count_examples": np.asarray(c_0, dtype="float64"),
        "rating": np.asarray(ratings, dtype="int64"),
    }


with pm.Model() as model:
    n_examples = pm.Data("n_examples", np.ones(1, dtype="float64"))
    count_target = pm.Data("count_target", np.ones((1, 100), dtype="float64"))
    count_examples = pm.Data("count_examples", np.ones((1, 100), dtype="float64"))

    # Typical preferred range width (Erlang scale) and lapse rate.
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=0.5)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
    lapse = pm.Beta("lapse", alpha=2.0, beta=2.0)

    s = pt.as_tensor_variable(SIZES)
    # log prior (Erlang, shape 2) + log size-principle likelihood, per size.
    logw = (pt.log(s)[None, :] - s[None, :] / sigma
            - n_examples[:, None] * pt.log(s)[None, :])
    logw = logw - pt.max(logw, axis=1, keepdims=True)
    w = pt.exp(logw)
    num = pt.sum(w * count_target, axis=1)
    den = pt.sum(w * count_examples, axis=1)
    post = num / den
    p = (1.0 - lapse) * post + lapse * 0.5
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
