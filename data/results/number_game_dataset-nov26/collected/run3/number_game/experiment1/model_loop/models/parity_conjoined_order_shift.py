"""Refinement of monotone_order_range_prior_shift: parity-conjoined concepts.

People do the same tempered, serial-position-weighted Bayesian concept
learning over 45 individually weighted rules and all intervals of 1..100,
with a sorted list of >=3 examples shifting the prior log-odds from rules
toward intervals by kappa. The one change (taken from
parity_restricted_concepts) is that parity is conjoined with any concept:
people also entertain the odd-only and even-only version of every rule and
interval, which share an inferred prior fraction rho (rho = 0 recovers the
incumbent). Since those concepts are half the size, the size principle
favours them whenever all examples share a parity, so opposite-parity
targets, in range or not, are rejected more firmly. "Yes" is the posterior
probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_N = 4


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


def log_lik_table(membership):
    size = membership.sum(axis=1, keepdims=True)
    in_part = np.where(size > 0, ALPHA * membership / np.maximum(size, 1), 0.0)
    return np.log(in_part + (1 - ALPHA) / 100)


def _dedupe(rows):
    uniq, counts = np.unique(rows, axis=0, return_counts=True)
    return uniq, counts.astype("float64")


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
BASE_RULE = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
BASE_INT = np.array([(NUMBERS >= m) & (NUMBERS <= n) for m in NUMBERS for n in range(m, 101)], dtype="float64")
N_INT = BASE_INT.shape[0]
ODD = (NUMBERS % 2 == 1).astype("float64")
EVEN = 1.0 - ODD
# Families: plain, odd-only, even-only.
RULE_MEMBERSHIP = np.concatenate([BASE_RULE, BASE_RULE * ODD, BASE_RULE * EVEN])
_ODD_INT, _ODD_CNT = _dedupe(BASE_INT * ODD)
_EVEN_INT, _EVEN_CNT = _dedupe(BASE_INT * EVEN)
INT_MEMBERSHIP = np.concatenate([BASE_INT, _ODD_INT, _EVEN_INT])
INT_LOG_COUNT = np.log(np.concatenate([np.ones(N_INT), _ODD_CNT, _EVEN_CNT]))
INT_FAMILY = np.concatenate([np.zeros(N_INT), np.ones(len(_ODD_CNT)), np.full(len(_EVEN_CNT), 2)]).astype("int64")
N_RULE_ALL = RULE_MEMBERSHIP.shape[0]
MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INT_MEMBERSHIP])
EXAMPLE_LOG_LIK = log_lik_table(MEMBERSHIP)  # (concepts, 100)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    pos_ll = pm.Data("pos_ll", np.zeros((1, MAX_N, MEMBERSHIP.shape[0])))
    pos_rel = pm.Data("pos_rel", np.zeros((1, MAX_N)))   # -0.5 .. +0.5 serial position
    pos_mask = pm.Data("pos_mask", np.ones((1, MAX_N)))  # 1 for real examples
    sorted_order = pm.Data("sorted_order", np.zeros(1))  # 1 if >=3 examples shown sorted
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    log_beta = pm.Normal("log_beta", mu=0.0, sigma=0.75)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)   # serial-position gradient
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)   # sorted-order shift rules -> intervals
    # The one change: prior share of parity-restricted concepts.
    rho = pm.Beta("rho", 1.0, 3.0)
    fam = pt.stack([pt.log1p(-rho), pt.log(rho / 2), pt.log(rho / 2)])

    rule_logodds = pt.log(lambda_0) - pt.log1p(-lambda_0) - kappa * sorted_order  # (sets,)
    log_rule_share = -pt.softplus(-rule_logodds)
    log_int_share = -pt.softplus(rule_logodds)

    raw = pt.exp(gamma * pos_rel) * pos_mask
    n_ex = pos_mask.sum(axis=1, keepdims=True)
    pos_w = beta * raw * n_ex / raw.sum(axis=1, keepdims=True)
    ll = (pos_w[:, :, None] * pos_ll).sum(axis=1)  # (sets, concepts)

    rule_prior = (fam[:, None] + pt.log(rule_weights)[None, :]).reshape((-1,))
    int_prior = -np.log(N_INT) + fam[INT_FAMILY] + INT_LOG_COUNT
    log_joint = pt.concatenate([
        log_rule_share[:, None] + rule_prior[None, :] + ll[:, :N_RULE_ALL],
        log_int_share[:, None] + int_prior[None, :] + ll[:, N_RULE_ALL:],
    ], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = pt.dot(posterior, pt.as_tensor(MEMBERSHIP))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    pll = np.zeros((S, MAX_N, MEMBERSHIP.shape[0]))
    prel = np.zeros((S, MAX_N))
    pmask = np.zeros((S, MAX_N))
    sorted_order = np.zeros(S)
    for i, s in enumerate(unique):
        n = len(s)
        for k, x in enumerate(s):
            pll[i, k] = EXAMPLE_LOG_LIK[:, x - 1]
            pmask[i, k] = 1.0
            prel[i, k] = (k / (n - 1) - 0.5) if n > 1 else 0.0
        diffs = [b - a for a, b in zip(s, s[1:])]
        if n >= 3 and (all(x > 0 for x in diffs) or all(x < 0 for x in diffs)):
            sorted_order[i] = 1.0
    return {
        "pos_ll": pll,
        "pos_rel": prel,
        "pos_mask": pmask,
        "sorted_order": sorted_order,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
