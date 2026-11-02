"""Magnitude-biased example sampling (a refinement of extended_multiples_concepts).

People do the same Bayesian concept learning as extended_multiples_concepts (45
independently weighted rules plus a multiples-of-13..50 family taking an inferred
share `ext_share`, intervals of 1..100 with an Erlang(2, sigma) size prior, an
inferred share `noise` of random examples, examples drawn without replacement),
but they believe the program picks its in-concept examples with a bias by
magnitude: member x is chosen with weight (x / 100) ** bias, sequentially
without replacement. With bias > 0 small members are rarely shown, so concepts
reaching below the smallest example cost little and "yes" extends further
downward than upward; bias = 0 recovers the incumbent. "Yes" is the posterior
probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4


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


_RULES = rule_concepts()
N_BASE = len(_RULES)
EXT_M = list(range(13, 51))  # less-familiar multiples rules
N_EXT = len(EXT_M)
_RULES.update({f"multiples of {m}": list(range(m, 101, m)) for m in EXT_M})
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")

# Every interval [m, n], 1 <= m <= n <= 100.
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
N_INT = len(INT_LO)
INT_SIZE = (INT_HI - INT_LO + 1).astype("float64")
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")
ALL_MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])  # (concepts, 100)
LOG_MAG = np.log(NUMBERS / 100.0)
SUBSET_BITS = np.array([[(b >> k) & 1 for k in range(MAX_EX - 1)] for b in range(8)], dtype="float64")

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_BASE - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    # Rules: dense (sets, slots, rules) membership of each example.
    ex_idx = pm.Data("ex_idx", np.zeros((1, MAX_EX), dtype="int64"))  # (sets, slots): example - 1
    rule_in = pm.Data("rule_in", np.zeros((MAX_EX, 1, N_RULES)))  # (slots, sets, rules)
    # Intervals: one entry per (set, interval, example inside it).
    e_cell = pm.Data("e_cell", np.zeros(1, dtype="int64"))  # set * N_INT + interval
    e_int = pm.Data("e_int", np.zeros(1, dtype="int64"))  # interval
    e_ex = pm.Data("e_ex", np.zeros(1, dtype="int64"))  # example - 1
    e_sub = pm.Data("e_sub", np.zeros(1, dtype="int64"))  # set * 8 + bitmask of earlier examples inside
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)  # share of examples believed random
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    ext_share = pm.Beta("ext_share", 1.0, 4.0)
    all_weights = pt.concatenate([(1 - ext_share) * rule_weights, pt.ones(N_EXT) * ext_share / N_EXT])

    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_int_prior = pt.log(INT_SIZE) - INT_SIZE / sigma
    log_int_prior = log_int_prior - pt.logsumexp(log_int_prior)
    log_prior = pt.concatenate([pt.log(lambda_0 * all_weights), pt.log(1 - lambda_0) + log_int_prior])

    # The one change: in-concept examples are picked one by one without
    # replacement, member x with weight (x / 100) ** bias.
    bias = pm.Normal("bias", 0.0, 1.0)
    w = pt.exp(bias * LOG_MAG)  # (100,)
    odds = (1 - noise) / (noise / 100.0)  # likelihoods relative to a random example
    rule_w = pt.dot(RULE_MEMBERSHIP, w)  # (rules,) total weight of each rule
    rule_lik = 0.0
    shown = 0.0  # weight of the rule's members already shown
    for j in range(MAX_EX):
        w_j = w[ex_idx[:, j]][:, None]  # padded slots have rule_in = 0
        rule_lik = rule_lik + rule_in[j] * pt.log1p(odds * w_j / (rule_w[None, :] - shown + 1e-9))
        shown = shown + rule_in[j] * w_j
    int_w = pt.dot(INTERVAL_MEMBERSHIP, w)  # (intervals,)
    subset_shown = pt.dot(w[ex_idx[:, :MAX_EX - 1]], SUBSET_BITS.T)  # (sets, 8) weight of shown subsets
    e_shown = subset_shown.reshape((-1,))[e_sub]
    e_term = pt.log1p(odds * w[e_ex] / (int_w[e_int] - e_shown + 1e-9))
    n_sets = ex_idx.shape[0]
    int_lik = pt.inc_subtensor(pt.zeros(n_sets * N_INT)[e_cell], e_term).reshape((n_sets, N_INT))
    log_lik = pt.concatenate([rule_lik, int_lik], axis=1)
    post = pt.special.softmax(log_prior[None, :] + log_lik, axis=1)
    in_concept = pt.dot(post, ALL_MEMBERSHIP)  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    ex_idx = np.zeros((len(unique), MAX_EX), dtype="int64")
    rule_in = np.zeros((MAX_EX, len(unique), N_RULES))
    cell, interval, example, sub = [], [], [], []
    for i, s in enumerate(unique):
        for j, x in enumerate(s[:MAX_EX]):
            ex_idx[i, j] = x - 1
            rule_in[j, i] = RULE_MEMBERSHIP[:, x - 1]
            inside = np.flatnonzero((INT_LO <= x) & (INT_HI >= x))
            bits = np.zeros(len(inside), dtype="int64")
            for k, y in enumerate(s[:j]):
                bits += ((INT_LO[inside] <= y) & (INT_HI[inside] >= y)).astype("int64") << k
            cell.append(i * N_INT + inside)
            interval.append(inside)
            example.append(np.full(len(inside), x - 1))
            sub.append(i * 8 + bits)
    return {
        "ex_idx": ex_idx,
        "rule_in": rule_in,
        "e_cell": np.concatenate(cell).astype("int64"),
        "e_int": np.concatenate(interval).astype("int64"),
        "e_ex": np.concatenate(example).astype("int64"),
        "e_sub": np.concatenate(sub).astype("int64"),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
