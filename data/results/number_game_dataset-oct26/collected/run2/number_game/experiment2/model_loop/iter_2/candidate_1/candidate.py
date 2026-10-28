"""Magnitude-range inference on a logarithmic mental number line.

People do not look for arithmetic rules or compare the target with each
example; they infer the range of magnitudes the program produces, reasoning
about size on a compressed, logarithmic mental number line. They favour the
narrowest range (in ratio terms) that covers all the examples, more strongly
the more examples there are (the size principle), and say "yes" in proportion
to their belief that the target falls inside that range — so generalization
extends further above the largest example than below the smallest, and tight
sets generalize narrowly while widely spread sets generalize broadly.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
MAXK = 4

# Log-scale width of every integer interval [a, b] of 1..100 (a <= b).
_a = np.arange(1, N + 1)[:, None].astype(float)
_b = np.arange(1, N + 1)[None, :].astype(float)
VALID = (_a <= _b).astype(float)
LOGW = np.where(VALID > 0, np.log(_b + 0.5) - np.log(_a - 0.5), 1.0)
NVEC = np.arange(1, MAXK + 1, dtype=float)


def prepare_observed(rows):
    n = len(rows)
    k_idx = np.zeros(n, dtype="int64")
    num_a = np.zeros(n, dtype="int64")
    num_b = np.zeros(n, dtype="int64")
    den_a = np.zeros(n, dtype="int64")
    den_b = np.zeros(n, dtype="int64")
    rating = np.zeros(n, dtype="int64")
    for r, row in enumerate(rows):
        ex = [int(x) for x in str(row["number_set"]).split(",")][:MAXK]
        t = int(row["target"])
        lo, hi = min(ex), max(ex)
        k_idx[r] = len(ex) - 1
        den_a[r], den_b[r] = lo - 1, hi - 1
        num_a[r], num_b[r] = min(lo, t) - 1, max(hi, t) - 1
        rv = row.get("rating")
        rating[r] = int(float(rv)) if rv is not None and str(rv).strip() != "" else 0
    return {"k_idx": k_idx, "num_a": num_a, "num_b": num_b,
            "den_a": den_a, "den_b": den_b, "rating": rating}


with pm.Model() as model:
    k_idx = pm.Data("k_idx", np.zeros(1, dtype="int64"))
    num_a = pm.Data("num_a", np.zeros(1, dtype="int64"))
    num_b = pm.Data("num_b", np.zeros(1, dtype="int64"))
    den_a = pm.Data("den_a", np.zeros(1, dtype="int64"))
    den_b = pm.Data("den_b", np.zeros(1, dtype="int64"))

    # Prior scale of range widths (in log units) and strength of the size principle.
    width_scale = pm.LogNormal("width_scale", mu=0.0, sigma=1.0)
    size_strength = pm.LogNormal("size_strength", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=-2.0, sigma=1.5)
    gain = pm.HalfNormal("gain", sigma=5.0)

    logw = pt.as_tensor_variable(LOGW)
    valid = pt.as_tensor_variable(VALID)
    # Erlang(2) prior on log width times size-principle likelihood width^-(n*strength).
    logweight = (pt.log(logw)[None] * (1.0 - size_strength * NVEC[:, None, None])
                 - logw[None] / width_scale)
    logweight = logweight - pt.max(logweight, axis=(1, 2), keepdims=True)
    weight = pt.exp(logweight) * valid[None]
    # C[n, A, B] = total weight of ranges with a <= A and b >= B.
    cum = pt.cumsum(weight, axis=1)
    cum = pt.cumsum(cum[:, :, ::-1], axis=2)[:, :, ::-1]
    num = cum[k_idx, num_a, num_b]
    den = cum[k_idx, den_a, den_b]
    belief = num / den

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + gain * belief), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
