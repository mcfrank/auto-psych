"""PyMC adapter for the Griffiths et al. (2018) motif-HMM model family.

randomness(x) = log P(x|fair) - log P(x|regular) with the regular process
their six-state, four-motif HMM (Eq. 7-9), row-normalised per footnote 11 and
marginalised over hidden state paths via an unrolled forward pass (Eq. 8) —
the two respects in which ``bayesian_diagnosticity`` is only approximate. The
sequences enter through the per-symbol columns sym1..sym8 (H=1, zero-padded
past n); steps beyond a trial's length are masked out. See the pure-Python
twin in ``model_families/motif_hmm.py`` for the full rationale.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_SEQ_LEN = 8

# Even states emit H, odd states emit T.
_EVEN_STATES = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def _matrices(delta, alpha):
    """Row-normalised transition matrix and initial vector (Eq. 9 + fn. 11)."""
    a, a2, d = alpha, alpha**2, delta
    zero = pt.zeros_like(a)
    rows = pt.stack(
        [
            pt.stack([d, a, a2, zero, zero, a2]),
            pt.stack([a, d, a2, zero, zero, a2]),
            pt.stack([a, a, zero, d, zero, a2]),
            pt.stack([a, a, d, zero, zero, a2]),
            pt.stack([a, a, a2, zero, zero, d]),
            pt.stack([a, a, a2, zero, d, zero]),
        ]
    )
    transition = rows / rows.sum(axis=1, keepdims=True)
    init_raw = pt.stack([a, a, a2, zero, zero, a2])
    init = init_raw / init_raw.sum()
    return init, transition


def _log_p_regular(n, syms, init, transition):
    """log P(x | motif HMM) per trial, forward pass unrolled over MAX_SEQ_LEN."""
    sym_f = [pt.cast(s, "float64") for s in syms]
    emission = [
        s[:, None] * _EVEN_STATES[None, :] + (1.0 - s)[:, None] * (1.0 - _EVEN_STATES)[None, :]
        for s in sym_f
    ]
    forward = init[None, :] * emission[0]
    for t in range(2, MAX_SEQ_LEN + 1):
        stepped = pt.dot(forward, transition) * emission[t - 1]
        active = pt.ge(n, t)[:, None]
        forward = pt.switch(active, stepped, forward)
    return pt.log(forward.sum(axis=1))


with pm.Model() as model:
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    syms_a = [
        pm.Data(f"sym{i}_a", np.zeros(1, dtype="int64"))
        for i in range(1, MAX_SEQ_LEN + 1)
    ]
    syms_b = [
        pm.Data(f"sym{i}_b", np.zeros(1, dtype="int64"))
        for i in range(1, MAX_SEQ_LEN + 1)
    ]
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    delta = pm.Uniform("delta", lower=0.01, upper=0.99)
    alpha = pm.Uniform("alpha", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    init, transition = _matrices(delta, alpha)

    log_fair_a = pt.cast(n_a, "float64") * np.log(0.5)
    log_fair_b = pt.cast(n_b, "float64") * np.log(0.5)
    score_a = log_fair_a - _log_p_regular(n_a, syms_a, init, transition)
    score_b = log_fair_b - _log_p_regular(n_b, syms_b, init, transition)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
