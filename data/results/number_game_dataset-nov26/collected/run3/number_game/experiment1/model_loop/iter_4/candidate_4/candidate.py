"""Refinement of monotone_order_range_prior_shift: parity-restricted concepts.

People do the same tempered, serial-position-weighted Bayesian concept
learning over 45 individually weighted rules and all intervals of 1..100,
with the same sorted-order shift (kappa) of prior log-odds from rules toward
intervals for sorted lists of >= 3 examples. The one change (taken from
parity_restricted_concepts): parity is conjoined with any concept, so the
odd-only and even-only version of every rule and interval is also
entertained, with an inferred prior share rho (rho = 0 recovers the
incumbent). Being half the size, a parity-restricted concept is favoured by
the size principle when all examples share a parity, so opposite-parity
rule members (e.g. 72, 84, 96 after 69, 9, 39, 21) are rejected. "Yes" is the
posterior probability that the concept contains the target.
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
RULE_MEMBERSHIP = np.concatenate([BASE_RULE, BASE_RULE * ODD, BASE_RULE * EVEN])  # (3*45, 100)
_ODD_INT, _ODD_CNT = _dedupe(BASE_INT * ODD)
_EVEN_INT, _EVEN_CNT = _dedupe(BASE_INT * EVEN)
INT_MEMBERSHIP = np.concatenate([BASE_INT, _ODD_INT, _EVEN_INT])
INT_LOG_COUNT = np.log(np.concatenate([np.ones(N_INT), _ODD_CNT, _EVEN_CNT]))
INT_FAMILY = np.concatenate([np.zeros(N_INT), np.ones(len(_ODD_CNT)), np.full(len(_EVEN_CNT), 2)]).astype("int64")
N_INT_ALL = INT_MEMBERSHIP.shape[0]
RULE_LL = log_lik_table(RULE_MEMBERSHIP)
INT_LL = log_lik_table(INT_MEMBERSHIP)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_pos_ll = pm.Data("rule_pos_ll", np.zeros((1, MAX_N, 3 * N_RULES)))
    interval_pos_ll = pm.Data("interval_pos_ll", np.zeros((1, MAX_N, N_INT_ALL)))
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
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)   # sorted-order shift toward intervals
    # The one change: prior share of parity-restricted concepts (odd / even halves).
    rho = pm.Beta("rho", 1.0, 3.0)
    fam = pt.stack([pt.log1p(-rho), pt.log(rho / 2), pt.log(rho / 2)])

    rule_logodds = pt.log(lambda_0) - pt.log1p(-lambda_0) - kappa * sorted_order  # (sets,)
    log_rule_share = -pt.softplus(-rule_logodds)
    log_int_share = -pt.softplus(rule_logodds)

    raw = pt.exp(gamma * pos_rel) * pos_mask
    n_ex = pos_mask.sum(axis=1, keepdims=True)
    pos_w = beta * raw * n_ex / raw.sum(axis=1, keepdims=True)  # (sets, MAX_N)

    rule_ll = (pos_w[:, :, None] * rule_pos_ll).sum(axis=1)       # (sets, 3*N_RULES)
    int_ll = (pos_w[:, :, None] * interval_pos_ll).sum(axis=1)    # (sets, N_INT_ALL)

    log_rule_prior = (fam[:, None] + pt.log(rule_weights)[None, :]).reshape((-1,))
    log_int_prior = fam[INT_FAMILY] + INT_LOG_COUNT - np.log(N_INT)
    log_joint = pt.concatenate([
        log_rule_share[:, None] + log_rule_prior[None, :] + rule_ll,
        log_int_share[:, None] + log_int_prior[None, :] + int_ll,
    ], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    membership = pt.as_tensor(np.concatenate([RULE_MEMBERSHIP, INT_MEMBERSHIP]))
    in_concept = pt.dot(posterior, membership)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_pos = np.zeros((S, MAX_N, 3 * N_RULES))
    int_pos = np.zeros((S, MAX_N, N_INT_ALL))
    pos_rel = np.zeros((S, MAX_N))
    pos_mask = np.zeros((S, MAX_N))
    sorted_order = np.zeros(S)
    for i, s in enumerate(unique):
        n = len(s)
        for k, x in enumerate(s):
            rule_pos[i, k] = RULE_LL[:, x - 1]
            int_pos[i, k] = INT_LL[:, x - 1]
            pos_mask[i, k] = 1.0
            pos_rel[i, k] = (k / (n - 1) - 0.5) if n > 1 else 0.0
        diffs = [b - a for a, b in zip(s, s[1:])]
        if n >= 3 and (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
            sorted_order[i] = 1.0
    return {
        "rule_pos_ll": rule_pos,
        "interval_pos_ll": int_pos,
        "pos_rel": pos_rel,
        "pos_mask": pos_mask,
        "sorted_order": sorted_order,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
