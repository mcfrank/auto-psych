"""Nearest-exemplar feature similarity with a Luce criterion
(refinement of exemplar_feature_similarity_gcm).

Same exemplar representation as the GCM incumbent: each number is coded by
its magnitude and its arithmetic features (parity, divisibility by
3/4/5/10, final digit, tens digit, prime, square, power of 2), and
similarity decays exponentially with distance weighted on each dimension.
The one change: the target is judged by its similarity to the single MOST
similar example (nearest-neighbour retrieval), compared against a criterion
by Luce's ratio rule, p(yes) = s / (s + k), instead of summed similarity, so
set size does not inflate "yes" and a target matching any one example is
accepted regardless of the others.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_PRIMES = {n for n in range(2, 101) if all(n % k for k in range(2, int(n ** 0.5) + 1))}
_SQUARES = {k * k for k in range(1, 11)}
_POW2 = {2 ** k for k in range(0, 7)}


def _feats(n):
    return [
        n % 2 == 0, n % 3 == 0, n % 4 == 0, n % 5 == 0, n % 10 == 0,
        n in _PRIMES, n in _SQUARES, n in _POW2,
    ]


def _mismatch(a, b):
    fa, fb = _feats(a), _feats(b)
    m = sum(x != y for x, y in zip(fa, fb))
    m += (a % 10) != (b % 10)
    m += (a // 10) != (b // 10)
    return m / 10.0


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    out = {}
    for i in range(4):
        if i < len(xs):
            out[f"dmag{i}"] = abs(t - xs[i]) / 100.0
            out[f"dfeat{i}"] = _mismatch(t, xs[i])
            out[f"mask{i}"] = 1.0
        else:
            out[f"dmag{i}"] = 0.0
            out[f"dfeat{i}"] = 0.0
            out[f"mask{i}"] = 0.0
    return out


with pm.Model() as model:
    dmag = pt.stack([pm.Data(f"dmag{i}", np.zeros(1, dtype="float64")) for i in range(4)], axis=1)
    dfeat = pt.stack([pm.Data(f"dfeat{i}", np.zeros(1, dtype="float64")) for i in range(4)], axis=1)
    mask = pt.stack([pm.Data(f"mask{i}", np.zeros(1, dtype="float64")) for i in range(4)], axis=1)

    # Sensitivity to magnitude and to arithmetic-feature mismatch
    # (separate rates, so no sensitivity/attention trade-off).
    c_mag = pm.LogNormal("c_mag", mu=1.0, sigma=1.0)
    c_feat = pm.LogNormal("c_feat", mu=0.5, sigma=1.0)
    # Log Luce criterion: the log-similarity at which yes and no are equal.
    log_k = pm.Normal("log_k", mu=-1.0, sigma=1.5)

    dist = c_mag * dmag + c_feat * dfeat
    # Absent example slots get a huge distance, so they never are the nearest.
    nn_dist = pt.min(dist + (1.0 - mask) * 1e3, axis=1)
    # Luce ratio rule s / (s + k) with s = exp(-nn_dist): sigmoid(-nn_dist - log k).
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(-nn_dist - log_k))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
