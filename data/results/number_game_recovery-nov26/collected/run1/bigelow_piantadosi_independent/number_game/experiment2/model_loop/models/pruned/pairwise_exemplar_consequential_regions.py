"""Exemplar-based generalisation by pairwise consequential-region similarity.

People store each example as a separate exemplar and compare the target with
each one in turn: the similarity of target t to example e is Shepard/Tenenbaum's
single-example generalisation probability, the posterior probability (from e
alone) that a consequential region containing e also contains t, over the 45
arithmetic/digit rules (prior share lambda, split equally) and all intervals of
1..100 (Erlang prior on width, scale sigma), each region weighted by 1/|region|.
The yes probability is the average similarity across the stored examples (no
joint inference, no exception discounting), passed through a small lapse.
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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# Rule similarity mass for every pair (e, t): sum over rules holding both of 1/|h|.
RULE_PAIR = (RULE_MEMBERSHIP / RULE_SIZE[:, None]).T @ RULE_MEMBERSHIP  # (100, 100)
SIZES = np.arange(1, 101, dtype="float64")
SIZE_COUNT = 101.0 - SIZES  # number of intervals of each width
MAX_EX = 4


def _interval_counts(e, t):
    """Number of intervals of each width (1..100) containing both e and t."""
    lo, hi = min(e, t), max(e, t)
    s = SIZES.astype(int)
    # start a in [max(1, hi - s + 1), min(lo, 101 - s)]
    first = np.maximum(1, hi - s + 1)
    last = np.minimum(lo, 101 - s)
    return np.maximum(0, last - first + 1).astype("float64")


def prepare_observed(rows):
    pairs = {}

    def pid(e, t):
        if (e, t) not in pairs:
            pairs[(e, t)] = len(pairs)
        return pairs[(e, t)]

    n = len(rows)
    idx_pair = np.zeros((n, MAX_EX), dtype="int64")
    idx_self = np.zeros((n, MAX_EX), dtype="int64")
    mask = np.zeros((n, MAX_EX), dtype="float64")
    rating = np.zeros(n, dtype="int64")
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        for j, e in enumerate(ex[:MAX_EX]):
            idx_pair[i, j] = pid(e, t)
            idx_self[i, j] = pid(e, e)
            mask[i, j] = 1.0
        rating[i] = int(float(r["rating"]))
    keys = list(pairs.keys())
    rule_pair = np.array([RULE_PAIR[e - 1, t - 1] for e, t in keys])
    int_counts = np.array([_interval_counts(e, t) for e, t in keys])
    return {
        "pair_rule": rule_pair,
        "pair_int_counts": int_counts,
        "idx_pair": idx_pair,
        "idx_self": idx_self,
        "ex_mask": mask,
        "rating": rating,
    }


with pm.Model() as model:
    pair_rule = pm.Data("pair_rule", np.zeros(1))
    pair_int_counts = pm.Data("pair_int_counts", np.zeros((1, 100)))
    idx_pair = pm.Data("idx_pair", np.zeros((1, MAX_EX), dtype="int64"))
    idx_self = pm.Data("idx_self", np.zeros((1, MAX_EX), dtype="int64"))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAX_EX)))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lam = pm.Beta("lambda_rule", 2.0, 2.0)  # prior share of rules
    sigma = pm.LogNormal("sigma", mu=np.log(15.0), sigma=0.8)  # typical interval width
    lapse = pm.Beta("lapse", 1.0, 9.0)
    base = pm.Beta("base", 2.0, 2.0)

    # Erlang prior per interval of width s, normalised over all intervals.
    log_erl = pt.log(SIZES) - SIZES / sigma
    log_z = pt.logsumexp(log_erl + np.log(SIZE_COUNT))
    w = pt.exp(log_erl - log_z) / SIZES  # prior x size-principle weight, per width
    mass = lam / N_RULES * pair_rule + (1.0 - lam) * pt.dot(pair_int_counts, w)

    sim = mass[idx_pair] / mass[idx_self]  # (trials, slots), in (0, 1]
    g = pt.sum(sim * ex_mask, axis=1) / pt.sum(ex_mask, axis=1)
    p = (1.0 - lapse) * g + lapse * base
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
