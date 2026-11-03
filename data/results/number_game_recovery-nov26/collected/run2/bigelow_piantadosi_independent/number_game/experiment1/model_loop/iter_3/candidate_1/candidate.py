"""Recency-weighted category-feature exemplar similarity.

People do not infer a hidden concept; they judge a target by its summed
similarity to the remembered examples (an exemplar account), where two numbers
are similar to the extent they are close in magnitude and share membership in
stored number categories (even, odd, primes, squares, powers, multiples, final
digit, contained digit), with small, distinctive categories counting more than
broad ones (each shared category weighted by 1 / its size). The examples are
held in memory with strengths that fade with how far back in the sequence they
were shown, so the most recent examples dominate the comparison.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = 4
NUMBERS = np.arange(1, 101)


def _categories():
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    cats = [
        [n for n in NUMBERS if n % 2 == 1],
        list(range(2, 101, 2)),
        primes,
        [n * n for n in range(1, 11)],
        [n**3 for n in range(1, 5)],
        [2, 4, 8, 16, 64],
        [2, 4, 8, 16, 32, 37, 64],
    ]
    cats += [list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    cats += [[m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)]
    cats += [[n for n in NUMBERS if n % 10 == d] for d in range(10)]
    cats += [[n for n in NUMBERS if str(d) in str(n)] for d in range(10)]
    return cats


_MEM = np.array([np.isin(NUMBERS, c) for c in _categories()], dtype="float64")  # (cats, 100)
_W = 1.0 / _MEM.sum(axis=1)  # distinctiveness weight of each category
_WMEM = _MEM * _W[:, None]
_SHARED = _WMEM.T @ _MEM  # [x, t] = weighted mass of x's categories also containing t
_OWN = _WMEM.sum(axis=0)  # weighted mass of all of x's categories
_FEAT_MISMATCH = 1.0 - _SHARED / _OWN[:, None]  # in [0, 1], 0 when t == x


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")][:K]
    t = int(target)
    n = len(xs)
    out = {}
    for i in range(K):
        if i < n:
            x = xs[i]
            out[f"mag_{i}"] = abs(x - t) / 10.0
            out[f"feat_{i}"] = float(_FEAT_MISMATCH[x - 1, t - 1])
            out[f"lag_{i}"] = float(n - 1 - i)  # 0 = last shown
            out[f"mask_{i}"] = 1.0
        else:
            out[f"mag_{i}"] = 0.0
            out[f"feat_{i}"] = 0.0
            out[f"lag_{i}"] = 0.0
            out[f"mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    mag = pt.stack([pm.Data(f"mag_{i}", np.zeros(1)) for i in range(K)], axis=1)
    feat = pt.stack([pm.Data(f"feat_{i}", np.zeros(1)) for i in range(K)], axis=1)
    lag = pt.stack([pm.Data(f"lag_{i}", np.zeros(1)) for i in range(K)], axis=1)
    mask = pt.stack([pm.Data(f"mask_{i}", np.zeros(1)) for i in range(K)], axis=1)

    # Attention to magnitude and to shared-category structure.
    w_mag = pm.HalfNormal("w_mag", sigma=2.0)
    w_feat = pm.HalfNormal("w_feat", sigma=5.0)
    # Memory retention per step back in the sequence (1 = no fading).
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    # Mapping from summed similarity to a yes response.
    bias = pm.Normal("bias", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    sim = pt.exp(-(w_mag * mag + w_feat * feat))
    strength = pt.exp(lag * pt.log(retention)) * mask
    mean_sim = pt.sum(strength * sim, axis=1) / pt.sum(strength, axis=1)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(bias + gain * mean_sim), 1e-6, 1 - 1e-6)
    )
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
