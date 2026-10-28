"""Arithmetic-operation generative program.

People imagine the hidden program as a generator that makes each new number
from one of the numbers it has already produced (most often its latest) by a
single elementary arithmetic operation: repeating it, rescaling it by a ratio
(small ratio changes far likelier than large ones), jumping by tens so the
final digit is kept, or moving to another number that shares a factor with
it. They answer "yes" in proportion to how probable it is that such a program
would produce the target, relative to an arbitrary number. Because rescaling
works on ratios, an equal linear gap is less likely among small numbers and
below the examples than among large numbers and above them.
"""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
N = 100
_v = np.arange(1, N + 1)
_LOGV = np.log(_v.astype("float64"))
# |log t - log x| for source x (row) and produced number t (column).
LOGDIST = np.abs(_LOGV[:, None] - _LOGV[None, :])
REPEAT = np.eye(N)
# Jump by tens: uniform over numbers 1-100 sharing x's final digit.
_same = (_v[:, None] % 10 == _v[None, :] % 10).astype("float64")
DIGIT = _same / _same.sum(axis=1, keepdims=True)
# Shared factor: probability grows with the log of the shared factor.
_fac = np.array([[math.log(math.gcd(int(x), int(t))) for t in _v] for x in _v])
_fac[0] = 1.0  # 1 shares no factor: spread uniformly
FACTOR = _fac / _fac.sum(axis=1, keepdims=True)


def prepare_observed(rows):
    n = len(rows)
    ex_idx = np.zeros((n, MAXK), dtype="int64")
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    tgt = np.zeros(n, dtype="int64")
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")][:MAXK]
        k = len(ex)
        for i, x in enumerate(ex):
            ex_idx[r, i] = min(max(x, 1), N) - 1
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        tgt[r] = min(max(int(row["target"]), 1), N) - 1
        rv = row.get("rating")
        rating[r] = int(float(rv)) if rv is not None and str(rv).strip() != "" else 0
    return {"ex_idx": ex_idx, "ex_lag": lag, "ex_mask": mask,
            "target_idx": tgt, "rating": rating}


with pm.Model() as model:
    ex_idx = pm.Data("ex_idx", np.zeros((1, MAXK), dtype="int64"))
    ex_lag = pm.Data("ex_lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    # Probability of each operation: repeat, rescale, jump by tens, factor.
    op_prob = pm.Dirichlet("op_prob", a=np.full(4, 2.0))
    # Typical log-ratio of a rescaling step (Laplace on the log number line).
    log_step = pm.Normal("log_step", mu=np.log(0.3), sigma=1.0)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=2.0)

    kern = pt.exp(-LOGDIST / pt.exp(log_step))
    scale_op = kern / pt.sum(kern, axis=1, keepdims=True)
    produce = (op_prob[0] * REPEAT + op_prob[1] * scale_op
               + op_prob[2] * DIGIT + op_prob[3] * FACTOR)

    p_from_ex = produce[ex_idx, target_idx[:, None]]
    mem = ex_mask * pt.exp(ex_lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    p_target = pt.sum(mem * p_from_ex, axis=1)

    logit = bias + gain * pt.log(N * p_target)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
