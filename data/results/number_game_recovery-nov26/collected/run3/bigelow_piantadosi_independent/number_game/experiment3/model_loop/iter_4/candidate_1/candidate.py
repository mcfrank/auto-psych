"""Background-normalized exemplar familiarity.

People do not infer a concept: they store the examples as exemplars and judge
a target by its similarity to them, with similarity falling off exponentially
with distance in a space of magnitude and familiar number properties (parity,
primality, squares, cubes, powers of 2, times tables 3-12, final digit, tens
digit). They say yes when the target is more similar to the stored examples
than to numbers in general (its mean similarity to all of 1-100), so a target
sharing a rare property with the examples stands out more than one sharing a
common property.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
_PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
           67, 71, 73, 79, 83, 89, 97}


def _props(n):
    v = [n % 2, n in _PRIMES, round(n ** 0.5) ** 2 == n,
         round(n ** (1 / 3)) ** 3 == n, (n & (n - 1)) == 0]
    v += [n % m == 0 for m in range(3, 13)]
    return np.array(v, dtype=float)


_P = np.stack([_props(n) for n in range(1, 101)])
_NDIM = _P.shape[1] + 2


def _ham(a, b):
    h = np.abs(_P[a - 1] - _P[b - 1]).sum()
    h += float(a % 10 != b % 10) + float(a // 10 != b // 10)
    return h / _NDIM


_NUMS = np.arange(1, 101)
BG_MAG = np.abs(_NUMS[:, None] - _NUMS[None, :]) / 100.0
BG_HAM = np.array([[_ham(a, b) for b in _NUMS] for a in _NUMS])


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")][:MAXK]
    t = int(target)
    f = {"tgt_idx": float(t - 1)}
    for k in range(MAXK):
        if k < len(ex):
            f[f"exm_mag_{k}"] = abs(t - ex[k]) / 100.0
            f[f"exm_ham_{k}"] = _ham(t, ex[k])
            f[f"exm_mask_{k}"] = 1.0
        else:
            f[f"exm_mag_{k}"] = 0.0
            f[f"exm_ham_{k}"] = 0.0
            f[f"exm_mask_{k}"] = 0.0
    return f


with pm.Model() as model:
    tgt_idx = pm.Data("tgt_idx", np.zeros(1, dtype="float64"))
    mags = pt.stack([pm.Data(f"exm_mag_{k}", np.zeros(1)) for k in range(MAXK)], axis=1)
    hams = pt.stack([pm.Data(f"exm_ham_{k}", np.zeros(1)) for k in range(MAXK)], axis=1)
    masks = pt.stack([pm.Data(f"exm_mask_{k}", np.zeros(1)) for k in range(MAXK)], axis=1)

    # specificity of the similarity gradient, attention to magnitude vs properties
    specificity = pm.LogNormal("specificity", mu=np.log(10.0), sigma=0.7)
    w_mag = pm.Beta("w_mag", alpha=2.0, beta=2.0)
    # response stage: criterion and gain on log relative familiarity
    criterion = pm.Normal("criterion", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=2.0)

    def sim(mag, ham):
        return pt.exp(-specificity * (w_mag * mag + (1.0 - w_mag) * ham))

    s_ex = (sim(mags, hams) * masks).sum(axis=1) / masks.sum(axis=1)
    s_bg_all = sim(pt.as_tensor(BG_MAG), pt.as_tensor(BG_HAM)).mean(axis=1)
    s_bg = s_bg_all[pt.cast(tgt_idx, "int64")]
    rel = pt.log(s_ex + 1e-12) - pt.log(s_bg)

    p = pm.math.sigmoid(criterion + gain * rel)
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
