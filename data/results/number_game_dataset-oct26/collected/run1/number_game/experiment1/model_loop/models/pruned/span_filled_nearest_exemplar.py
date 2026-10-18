"""Span-filled nearest-exemplar similarity with a Luce criterion
(refinement of nearest_exemplar_luce_similarity).

People judge the target by its similarity to the most similar example on
arithmetic features (parity, divisibility by 3/4/5/10, final digit, tens
digit, prime, square, power of 2), combined with magnitude similarity and
compared against a criterion by Luce's ratio rule. The one change: on the
magnitude dimension the gaps between examples are filled in, so magnitude
dissimilarity is the target's distance to the span [min example, max
example] (zero inside it) rather than to the nearest single example.
"""
import numpy as np
import pymc as pm

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
    lo, hi = min(xs), max(xs)
    span_dist = max(lo - t, 0, t - hi) / 100.0
    # Magnitude distance is shared by all examples (distance to the span), so
    # the nearest example is the one with the smallest feature mismatch.
    return {"span_dist": float(span_dist),
            "nn_feat": float(min(_mismatch(t, x) for x in xs))}


with pm.Model() as model:
    span_dist = pm.Data("span_dist", np.zeros(1, dtype="float64"))
    nn_feat = pm.Data("nn_feat", np.zeros(1, dtype="float64"))

    c_mag = pm.LogNormal("c_mag", mu=1.0, sigma=1.0)
    c_feat = pm.LogNormal("c_feat", mu=0.5, sigma=1.0)
    log_k = pm.Normal("log_k", mu=-1.0, sigma=1.5)

    dist = c_mag * span_dist + c_feat * nn_feat
    # Luce ratio rule s / (s + k) with s = exp(-dist).
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(-dist - log_k))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
