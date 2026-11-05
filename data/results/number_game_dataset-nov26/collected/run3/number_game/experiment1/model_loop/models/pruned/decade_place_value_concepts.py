"""Refinement of parity_restricted_concepts: decade (place-value) concepts.

People do the same tempered-evidence Bayesian concept learning over the 45
weighted rules, all intervals of 1..100, and their odd-only and even-only
versions, but they also read numbers by place value and entertain decade
concepts, "numbers whose tens digit is one of these" (e.g. the 20s, 30s or
60s), which share an inferred prior weight kappa (kappa = 0 recovers the
incumbent). The size principle favours the decades the examples occupy, so a
target breaking the examples' shared final digit is endorsed more when an
example sits in its decade (62,32,22 -> 25) than when none does (63,23,43 -> 35).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
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


def log_lik_table(membership):
    size = membership.sum(axis=1, keepdims=True)
    in_part = np.where(size > 0, ALPHA * membership / np.maximum(size, 1), 0.0)
    return np.log(in_part + (1 - ALPHA) / 100)


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
def _dedupe(rows):
    """Merge identical concepts; return distinct rows and their multiplicities."""
    uniq, counts = np.unique(rows, axis=0, return_counts=True)
    return uniq, counts.astype("float64")


# Parity-restricted intervals repeat (e.g. [2,8] and [3,7] odd-only are the
# same set); merging them with a log-multiplicity prior term is exact.
_ODD_INT, _ODD_CNT = _dedupe(BASE_INT * ODD)
_EVEN_INT, _EVEN_CNT = _dedupe(BASE_INT * EVEN)
INT_MEMBERSHIP = np.concatenate([BASE_INT, _ODD_INT, _EVEN_INT])
INT_LOG_COUNT = np.log(np.concatenate([np.ones(N_INT), _ODD_CNT, _EVEN_CNT]))
INT_FAMILY = np.concatenate([np.zeros(N_INT), np.ones(len(_ODD_CNT)), np.full(len(_EVEN_CNT), 2)]).astype("int64")
N_INT_ALL = INT_MEMBERSHIP.shape[0]
RULE_LL = log_lik_table(RULE_MEMBERSHIP)
INT_LL = log_lik_table(INT_MEMBERSHIP)

# Decade (place-value) concepts: every non-empty set of tens digits; 1-9 is
# the 0s decade and 100 is its own decade (11 decades, 2047 concepts).
DECADE = np.minimum(NUMBERS // 10, 10)
DEC_MEMBERSHIP = np.array([[(mask >> DECADE[i]) & 1 for i in range(100)] for mask in range(1, 2 ** 11)], dtype="float64")
N_DEC = DEC_MEMBERSHIP.shape[0]
DEC_LL = log_lik_table(DEC_MEMBERSHIP)


def _decade_groups(examples):
    """Group the decade concepts by their (identical) likelihood of the examples.

    Concepts in a group share prior and likelihood, so their posterior is equal;
    each group is summarised by its log-likelihood, log count and summed
    membership, which makes the decade family exact but cheap to evaluate.
    """
    ll = DEC_LL[:, np.array(examples) - 1].sum(axis=1)
    keys, inverse = np.unique(np.round(ll, 9), return_inverse=True)
    counts = np.bincount(inverse).astype("float64")
    member = np.zeros((len(keys), 100))
    np.add.at(member, inverse, DEC_MEMBERSHIP)
    return keys, np.log(counts), member / counts[:, None]

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, 3 * N_RULES)))
    interval_log_lik = pm.Data("interval_log_lik", np.zeros((1, N_INT_ALL)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    decade_log_lik = pm.Data("decade_log_lik", np.zeros((1, 1)))
    decade_log_count = pm.Data("decade_log_count", np.zeros((1, 1)))
    decade_member = pm.Data("decade_member", np.zeros((1, 1, 100)))
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
    # Prior share of parity-restricted concepts (split equally odd / even).
    rho = pm.Beta("rho", 1.0, 3.0)
    fam = pt.stack([pt.log1p(-rho), pt.log(rho / 2), pt.log(rho / 2)])  # (3,)

    # Prior share of decade concepts, spread uniformly over them.
    kappa = pm.Beta("kappa", 1.0, 9.0)

    log_rule_prior = (pt.log(lambda_0) + fam[:, None] + pt.log(rule_weights)[None, :]).reshape((-1,))
    log_int_prior = pt.log1p(-lambda_0) - np.log(N_INT) + fam[INT_FAMILY] + INT_LOG_COUNT
    log_rule_prior = log_rule_prior + pt.log1p(-kappa)
    log_int_prior = log_int_prior + pt.log1p(-kappa)
    log_dec_prior = pt.log(kappa) - np.log(N_DEC)
    log_joint = pt.concatenate([
        log_rule_prior[None, :] + beta * rule_log_lik,
        log_int_prior[None, :] + beta * interval_log_lik,
        log_dec_prior + decade_log_count + beta * decade_log_lik,
    ], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    n_ri = 3 * N_RULES + N_INT_ALL
    membership = pt.as_tensor(np.concatenate([RULE_MEMBERSHIP, INT_MEMBERSHIP]))
    in_concept = pt.dot(posterior[:, :n_ri], membership) + (
        posterior[:, n_ri:, None] * decade_member).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_ll = np.stack([RULE_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    int_ll = np.stack([INT_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    groups = [_decade_groups(s) for s in unique]
    g_max = max(len(g[0]) for g in groups)
    dec_ll = np.zeros((len(unique), g_max))
    dec_lc = np.full((len(unique), g_max), -1e4)  # padding groups get no weight
    dec_mem = np.zeros((len(unique), g_max, 100))
    for i, (k, lc, mem) in enumerate(groups):
        dec_ll[i, :len(k)], dec_lc[i, :len(k)], dec_mem[i, :len(k)] = k, lc, mem
    return {
        "rule_log_lik": rule_ll,
        "interval_log_lik": int_ll,
        "decade_log_lik": dec_ll,
        "decade_log_count": dec_lc,
        "decade_member": dec_mem,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
