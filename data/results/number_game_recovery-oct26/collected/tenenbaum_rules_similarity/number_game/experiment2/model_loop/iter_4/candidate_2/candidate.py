"""Attention-focused exemplar similarity.

People judge a target by its summed similarity to the remembered examples on a
multidimensional representation of numbers (magnitude, parity, final digit,
decade, divisibility by 3, 4 and 5, primality, squareness, power of two). The
metric is an attention-weighted city-block distance and similarity decays
exponentially with it. Attention goes to the dimensions on which the examples
agree with each other, in proportion to that agreement and by an inferred
degree of focusing, so a target that breaks a property the examples share is
judged far, while differences on dimensions the examples vary on barely matter.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4
PRIMES = {n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1))}
SQUARES = {k * k for k in range(1, 11)}
POW2 = {2 ** k for k in range(0, 7)}


def _features(n):
    """Categorical dimensions of a number (magnitude handled separately)."""
    return [
        n % 2,
        n % 10,
        min(n // 10, 9) if n < 100 else 10,
        int(n % 3 == 0),
        int(n % 4 == 0),
        int(n % 5 == 0),
        int(n in PRIMES),
        int(n in SQUARES),
        int(n in POW2),
    ]


N_CAT = 9
N_DIM = N_CAT + 1  # dimension 0 is magnitude


def _agreement(ex):
    """How consistent the examples are on each dimension (1 = all agree)."""
    if len(ex) < 2:
        return np.ones(N_DIM)
    pairs = list(itertools.combinations(ex, 2))
    agree = np.zeros(N_DIM)
    agree[0] = 1.0 - np.mean([abs(a - b) for a, b in pairs]) / 99.0
    for d in range(N_CAT):
        agree[d + 1] = np.mean([_features(a)[d] == _features(b)[d] for a, b in pairs])
    return agree


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    mism = np.zeros((len(cells), MAX_EX, N_DIM))
    mask = np.zeros((len(cells), MAX_EX))
    agree = np.zeros((len(cells), N_DIM))
    for i, (s, t) in enumerate(cells):
        ft = _features(t)
        agree[i] = _agreement(s)
        for k, x in enumerate(s):
            fx = _features(x)
            mism[i, k, 0] = abs(t - x) / 10.0
            mism[i, k, 1:] = [float(a != b) for a, b in zip(ft, fx)]
            mask[i, k] = 1.0
    return {
        "cell_mismatch": mism,
        "cell_mask": mask,
        "cell_agreement": agree,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    cell_mismatch = pm.Data("cell_mismatch", np.zeros((1, MAX_EX, N_DIM)))
    cell_mask = pm.Data("cell_mask", np.ones((1, MAX_EX)))
    cell_agreement = pm.Data("cell_agreement", np.ones((1, N_DIM)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Degree to which attention focuses on dimensions the examples share.
    focus = pm.HalfNormal("focus", sigma=5.0)
    # Baseline salience of magnitude relative to the categorical dimensions.
    magnitude_salience = pm.Normal("magnitude_salience", mu=0.0, sigma=2.0)
    # Steepness of the exponential similarity gradient.
    sensitivity = pm.LogNormal("sensitivity", mu=1.0, sigma=1.0)

    mag_offset = pt.concatenate([pt.stack([magnitude_salience]), pt.zeros(N_CAT)])
    attention = pt.special.softmax(focus * cell_agreement + mag_offset, axis=1)
    dist = (cell_mismatch * attention[:, None, :]).sum(axis=2)
    sim = (pt.exp(-sensitivity * dist) * cell_mask).sum(axis=1) / cell_mask.sum(axis=1)

    floor = pm.Beta("floor", alpha=1.0, beta=9.0)
    ceiling = pm.Beta("ceiling", alpha=9.0, beta=1.0)
    p_cell = floor + (ceiling - floor) * sim
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
