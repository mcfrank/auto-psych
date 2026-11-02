"""Spread-adaptive similarity on a compressed number line.

People judge whether the program would produce a target purely by how close it
lies to the examples on a logarithmically compressed mental number line, with an
exponential (Shepard-style) generalization gradient around each example whose
breadth is set by how spread out the examples are. Endorsement is the
probability that at least one example "captures" the target (noisy-or of the
per-example similarities); arithmetic and digit regularities play no role.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    feats = {"tgt": float(int(target))}
    for i in range(MAX_EX):
        feats[f"ex{i}"] = float(ex[i]) if i < len(ex) else float(ex[0])
        feats[f"mask{i}"] = 1.0 if i < len(ex) else 0.0
    return feats


with pm.Model() as model:
    tgt = pm.Data("tgt", np.zeros(1, dtype="float64"))
    exs = [pm.Data(f"ex{i}", np.ones(1, dtype="float64")) for i in range(MAX_EX)]
    masks = [pm.Data(f"mask{i}", np.ones(1, dtype="float64")) for i in range(MAX_EX)]

    # Compression offset: log(x + c); large c ~ linear line, small c ~ strong compression.
    log_c = pm.Normal("log_c", mu=np.log(10.0), sigma=1.0)
    c = pt.exp(log_c)
    # Gradient breadth (in compressed units): base width plus a share of the examples' spread.
    log_w0 = pm.Normal("log_w0", mu=np.log(0.1), sigma=1.0)
    k = pm.HalfNormal("k", sigma=1.0)
    # Response floor (yes to unsupported targets) and lapse at full similarity.
    floor = pm.Beta("floor", alpha=2.0, beta=8.0)
    lapse = pm.Beta("lapse", alpha=1.0, beta=15.0)

    E = pt.stack(exs, axis=0)          # (4, N)
    M = pt.stack(masks, axis=0)        # (4, N)
    pos_e = pt.log(E + c)
    pos_t = pt.log(tgt + c)
    big = 1e3
    hi = pt.max(pos_e - big * (1 - M), axis=0)
    lo = pt.min(pos_e + big * (1 - M), axis=0)
    spread = hi - lo
    width = pt.exp(log_w0) + k * spread

    sim = pt.exp(-pt.abs(pos_t[None, :] - pos_e) / width[None, :]) * M
    capture = 1.0 - pt.prod(1.0 - sim * 0.999999, axis=0)

    p = floor + (1.0 - lapse - floor) * capture
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
