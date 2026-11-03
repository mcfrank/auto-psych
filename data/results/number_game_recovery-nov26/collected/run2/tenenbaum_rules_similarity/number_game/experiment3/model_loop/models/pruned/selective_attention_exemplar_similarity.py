"""Selective-attention exemplar similarity (GCM with example-driven attention).

People do not infer a rule; they judge the target by its summed similarity to
the stored examples. Attention over number properties (odd/even, prime, square,
cube, powers, multiples of 3-12, final digit) is captured by properties the
examples share that are rare among 1-100; similarity to an example decays
exponentially with attention-weighted property mismatch plus a ratio-scale
(log) magnitude distance, and "yes" follows the Luce ratio of summed
similarity against a fixed criterion.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMS = np.arange(1, 101)
MAX_EX = 4


def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _powers(b):
    s, v = set(), b
    while v <= 100:
        s.add(v)
        v *= b
    return s


_PROPS = [lambda n: n % 2 == 1, lambda n: n % 2 == 0, _is_prime,
          lambda n: n in {i * i for i in range(1, 11)},
          lambda n: n in {i ** 3 for i in range(1, 5)}]
_PROPS += [(lambda s: (lambda n: n in s))(_powers(b)) for b in range(2, 11)]
_PROPS += [(lambda m: (lambda n: n % m == 0))(m) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
_PROPS += [(lambda d: (lambda n: n % 10 == d))(d) for d in range(10)]
FMAT = np.array([[1.0 if p(int(n)) else 0.0 for p in _PROPS] for n in NUMS])
N_F = FMAT.shape[1]
# Distinctiveness of sharing a property: rarer properties are more surprising.
RARITY = np.log(100.0 / FMAT.sum(axis=0))

SAMPLER_SETTINGS = {"target_accept": 0.9}


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    keys = [(s, int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    idx = {k: i for i, k in enumerate(cells)}
    C = len(cells)
    mism = np.zeros((C, MAX_EX, N_F))
    logratio = np.zeros((C, MAX_EX))
    mask = np.full((C, MAX_EX), -1e3)
    score = np.zeros((C, N_F))
    n_ex = np.zeros(C)
    for i, (s, t) in enumerate(cells):
        ex = np.array(s)
        score[i] = FMAT[ex - 1].mean(axis=0) * RARITY
        n_ex[i] = len(s)
        for j, x in enumerate(s):
            mism[i, j] = np.abs(FMAT[t - 1] - FMAT[x - 1])
            logratio[i, j] = abs(np.log(t / x))
            mask[i, j] = 0.0
    return {
        "mismatch": mism, "log_ratio": logratio, "ex_mask": mask,
        "attn_score": score, "log_n_ex": np.log(n_ex),
        "row_cell": np.array([idx[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, MAX_EX, N_F)))
    log_ratio = pm.Data("log_ratio", np.zeros((1, MAX_EX)))
    ex_mask = pm.Data("ex_mask", np.zeros((1, MAX_EX)))
    attn_score = pm.Data("attn_score", np.zeros((1, N_F)))
    log_n_ex = pm.Data("log_n_ex", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # How sharply attention concentrates on the examples' distinctive shared properties.
    attn_focus = pm.LogNormal("attn_focus", mu=np.log(2.0), sigma=0.75)
    # Specificity of property similarity and of ratio-scale magnitude similarity.
    specificity = pm.LogNormal("specificity", mu=np.log(5.0), sigma=1.0)
    magnitude_spec = pm.LogNormal("magnitude_spec", mu=np.log(2.0), sigma=1.0)
    # Luce criterion (background "no" strength), on the log scale.
    log_criterion = pm.Normal("log_criterion", mu=0.0, sigma=2.0)

    attention = pt.special.softmax(attn_focus * attn_score, axis=1)
    feat_dist = (mismatch * attention[:, None, :]).sum(axis=2)
    log_sim = -specificity * feat_dist - magnitude_spec * log_ratio + ex_mask
    log_summed = pt.logsumexp(log_sim, axis=1) - log_n_ex
    cell_p = pm.math.sigmoid(log_summed - log_criterion)
    p_yes = pm.Deterministic("p_yes", pt.clip(cell_p[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
