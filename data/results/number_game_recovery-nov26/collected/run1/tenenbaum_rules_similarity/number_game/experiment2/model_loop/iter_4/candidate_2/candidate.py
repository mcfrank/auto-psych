"""Conjunctive (geometric-mean) shared-category similarity.

People do not infer a program; they judge a target by its similarity to the
examples, where two numbers are similar to the degree that they fall together in
familiar number categories (parity, multiples, final or contained digit, primes,
squares, powers, magnitude ranges and thresholds), with small distinctive shared
categories counting more than broad ones (Shepard-style additive clustering, each
category weighted by a family salience and 1/size; bounded ranges also by an Erlang
prior on width). Similarity to each example is normalised by that example's
self-similarity, and the similarities are combined conjunctively, as a geometric
mean over the examples (mean log similarity): a target must resemble every example.
p(yes) is a logistic function of the mean log similarity.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("powers of 2,"):
        return 5
    return 1


_RULES = rule_concepts()
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in _RULES.values()], dtype="float64")
N_RULES = len(RULE_MEMBERSHIP)
# Distinctiveness-weighted co-membership of every (exemplar, target) pair in rules.
RULE_SHARED = RULE_MEMBERSHIP[:, :, None] * RULE_MEMBERSHIP[:, None, :] / RULE_MEMBERSHIP.sum(1)[:, None, None]

LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
THRESHOLD = (LO == 1) | (HI == 100)
IV = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 99)
N_PER_WIDTH = np.array([np.sum(~THRESHOLD & (WIDTH == w)) for w in WIDTHS], dtype="float64")
N_THRESHOLD = float(THRESHOLD.sum())
# Per width group: sum over ranges of [e in c][t in c] / |c|, shape (widths, 100, 100).
WIDTH_SHARED = np.stack(
    [(IV[sel].T @ IV[sel]) / w for w, sel in ((w, ~THRESHOLD & (WIDTH == w)) for w in WIDTHS)]
)
TH = IV[THRESHOLD]
THRESHOLD_SHARED = (TH.T / WIDTH[THRESHOLD]) @ TH


def prepare_observed(rows):
    n = len(rows)
    counts = np.zeros((n, 100))
    for i, r in enumerate(rows):
        for e in str(r["number_set"]).split(","):
            counts[i, int(e) - 1] += 1.0
    return {
        "example_count": counts,
        "target_idx": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    example_count = pm.Data("example_count", np.zeros((1, 100)))  # how often each number is an example
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    per_range = pt.exp(erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH)))

    rule_w = pt.exp(log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY]
    shared = (
        pt.tensordot(rule_w, RULE_SHARED, axes=1)
        + family_weight[6] * pt.tensordot(per_range, WIDTH_SHARED, axes=1)
        + family_weight[7] / N_THRESHOLD * THRESHOLD_SHARED
    )
    log_sim = pt.log(shared) - pt.log(pt.diagonal(shared))[:, None]  # (exemplar, target), <= 0

    total_log_sim = (pt.dot(example_count, log_sim) * pt.extra_ops.to_one_hot(target_idx, 100)).sum(axis=1)
    evidence = total_log_sim / example_count.sum(axis=1)
    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=2.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(bias + gain * evidence))
    pm.Bernoulli("response", p=p_yes, observed=rating)
