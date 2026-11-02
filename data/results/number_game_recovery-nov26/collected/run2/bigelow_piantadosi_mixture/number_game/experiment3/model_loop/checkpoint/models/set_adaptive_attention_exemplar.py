"""Set-adaptive attention exemplar similarity (GCM with attention to the set's shared dimensions).

People do not infer a hidden rule; they store the shown examples and say "yes"
to a target in proportion to its summed similarity to those stored exemplars.
Similarity falls off exponentially with the attention-weighted mismatch between
two numbers on a fixed set of number dimensions (magnitude, parity, residues mod
3, 4 and 5, last digit, tens digit, digit sum, shared digits, primality,
squareness, power-of-2-ness). Attention is allocated afresh for each set,
shifting toward the dimensions on which the examples agree with each other,
with the sharpness of that shift inferred.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4
SQUARES = {k * k for k in range(1, 11)}
POW2 = {1, 2, 4, 8, 16, 32, 64}


def _is_prime(n):
    return n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))


def _mismatch(a, b):
    """Per-dimension mismatch in [0, 1] between two numbers."""
    return [
        1.0 - np.exp(-abs(a - b) / 10.0),
        float(a % 2 != b % 2),
        float(a % 3 != b % 3),
        float(a % 4 != b % 4),
        float(a % 5 != b % 5),
        float(a % 10 != b % 10),
        float(a // 10 != b // 10),
        float(sum(map(int, str(a))) != sum(map(int, str(b)))),
        float(not (set(str(a)) & set(str(b)))),
        float(_is_prime(a) != _is_prime(b)),
        float((a in SQUARES) != (b in SQUARES)),
        float((a in POW2) != (b in POW2)),
    ]


N_DIM = 12


def prepare_observed(rows):
    n = len(rows)
    mism = np.zeros((n, MAX_EX, N_DIM))
    mask = np.zeros((n, MAX_EX))
    disp = np.zeros((n, N_DIM))
    cache = {}
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        for j, x in enumerate(ex[:MAX_EX]):
            mism[i, j] = _mismatch(t, x)
            mask[i, j] = 1.0
        key = tuple(ex)
        if key not in cache:
            pairs = list(itertools.combinations(ex, 2))
            cache[key] = (np.mean([_mismatch(a, b) for a, b in pairs], axis=0)
                          if pairs else np.zeros(N_DIM))
        disp[i] = cache[key]
    return {
        "mismatch": mism,
        "ex_mask": mask,
        "dispersion": disp,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}

with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, MAX_EX, N_DIM)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAX_EX)))
    dispersion = pm.Data("dispersion", np.zeros((1, N_DIM)))

    # How strongly attention shifts to dimensions the examples agree on.
    focus = pm.HalfNormal("focus", sigma=10.0)
    # Specificity of the exponential similarity gradient.
    specificity = pm.HalfNormal("specificity", sigma=10.0)
    # Response mapping from mean exemplar similarity to "yes".
    bias = pm.Normal("bias", mu=0.0, sigma=3.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    attention = pt.special.softmax(-focus * dispersion, axis=1)  # trials x dims
    dist = pt.sum(attention[:, None, :] * mismatch, axis=2)      # trials x examples
    sim = pt.exp(-specificity * dist) * ex_mask
    mean_sim = pt.sum(sim, axis=1) / pt.sum(ex_mask, axis=1)
    p = pm.math.sigmoid(bias + gain * (mean_sim - 0.5) * 10.0)
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
