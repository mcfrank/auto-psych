"""Prototype abstraction: people summarise the example set as one prototype — its
mean magnitude and the proportion of examples having each arithmetic property —
and say "yes" in proportion to the target's similarity to that prototype, which
decays exponentially with attention-weighted distance from it."""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

FEATS = ["even", "m3", "m4", "m5", "m10", "prime", "square", "pow2"]
_SQ = {i * i for i in range(1, 11)}
_P2 = {2 ** i for i in range(0, 7)}


def _is_prime(n):
    if n < 2:
        return False
    return all(n % d for d in range(2, int(math.isqrt(n)) + 1))


def _props(n):
    return {
        "even": float(n % 2 == 0), "m3": float(n % 3 == 0), "m4": float(n % 4 == 0),
        "m5": float(n % 5 == 0), "m10": float(n % 10 == 0), "prime": float(_is_prime(n)),
        "square": float(n in _SQ), "pow2": float(n in _P2),
    }


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")]
    t = int(target)
    tp = _props(t)
    eps = [_props(x) for x in ex]
    out = {"d_mag": abs(t - sum(ex) / len(ex)) / 100.0}
    for f in FEATS:
        proto = sum(p[f] for p in eps) / len(eps)
        out["d_" + f] = abs(tp[f] - proto)
    out["d_digit"] = 1.0 - sum(1.0 for x in ex if x % 10 == t % 10) / len(ex)
    return out


with pm.Model() as model:
    d_mag = pm.Data("d_mag", np.zeros(1, dtype="float64"))
    d_digit = pm.Data("d_digit", np.zeros(1, dtype="float64"))
    d_feats = [pm.Data("d_" + f, np.zeros(1, dtype="float64")) for f in FEATS]

    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    w_mag = pm.HalfNormal("w_mag", sigma=5.0)
    w_digit = pm.HalfNormal("w_digit", sigma=2.0)
    w_feat = pm.HalfNormal("w_feat", sigma=2.0, shape=len(FEATS))

    dist = w_mag * d_mag + w_digit * d_digit
    for i, d in enumerate(d_feats):
        dist = dist + w_feat[i] * d
    # Similarity to prototype (exp(-dist)) mapped to yes via a logistic criterion.
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept - dist))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
