"""Selective-attention exemplar similarity (GCM-style).

People do not infer a hidden concept; they judge a target by its average
similarity to the stored examples. Similarity decays exponentially with a
weighted city-block distance over the number's magnitude and its membership in
familiar number categories (the 45 categories: odd, even, primes, squares,
cubes, powers, multiples, final digit, contained digit). Attention is allocated
selectively by the examples: each category shared by all examples draws
salience |C|^-beta (beta fitted: >0 favours small, distinctive categories,
<0 broad, familiar ones such as parity), magnitude has a
fixed salience m, and attention weights are saliences normalised to sum to one.
The answer follows the Luce choice rule with a yes-criterion kappa:
p_yes = S / (S + kappa), S = mean similarity of the target to the examples.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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


_CATS = _categories()
N_CATS = len(_CATS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in _CATS], dtype="float64")  # (cats, 100)
LOG_CAT_SIZE = np.log(MEMBERSHIP.sum(axis=1))
MAX_EX = 4


def prepare_observed(rows):
    """Per trial: which categories all examples share, whether the target misses
    each, and the target's magnitude distance to each example (masked)."""
    n = len(rows)
    shared = np.zeros((n, N_CATS))
    miss = np.zeros((n, N_CATS))
    dist = np.zeros((n, MAX_EX))
    mask = np.zeros((n, MAX_EX))
    for j, r in enumerate(rows):
        ex = np.array([int(x) for x in str(r["number_set"]).split(",")])
        t = int(r["target"])
        shared[j] = MEMBERSHIP[:, ex - 1].min(axis=1)
        miss[j] = 1.0 - MEMBERSHIP[:, t - 1]
        dist[j, : len(ex)] = np.abs(t - ex) / 100.0
        mask[j, : len(ex)] = 1.0
    return {
        "shared_cat": shared,
        "target_miss": miss,
        "mag_dist": dist,
        "ex_mask": mask,
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    shared_cat = pm.Data("shared_cat", np.zeros((1, N_CATS)))
    target_miss = pm.Data("target_miss", np.zeros((1, N_CATS)))
    mag_dist = pm.Data("mag_dist", np.zeros((1, MAX_EX)))
    ex_mask = pm.Data("ex_mask", np.zeros((1, MAX_EX)))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    c = pm.LogNormal("c", mu=np.log(5.0), sigma=1.0)  # similarity sensitivity
    beta = pm.Normal("beta", mu=0.0, sigma=1.0)  # salience ~ size**-beta (beta>0: distinctive categories draw more)
    log_m = pm.Normal("log_m", mu=-2.0, sigma=2.0)  # salience of magnitude
    log_kappa = pm.Normal("log_kappa", mu=0.0, sigma=2.0)  # yes-criterion

    sal = shared_cat * pt.exp(-beta * LOG_CAT_SIZE)[None, :]  # (trials, cats)
    m = pt.exp(log_m)
    total = m + sal.sum(axis=1)
    cat_dist = (sal * target_miss).sum(axis=1) / total  # same for every example
    mag_w = m / total
    d = c * (cat_dist[:, None] + mag_w[:, None] * mag_dist)
    S = (ex_mask * pt.exp(-d)).sum(axis=1) / ex_mask.sum(axis=1)
    p = S / (S + pt.exp(log_kappa))

    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
