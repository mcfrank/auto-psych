"""Refinement of parity_restricted_concepts: a range needs two examples to bound it.

People do the same tempered-evidence Bayesian concept learning over the 45
weighted rules, all intervals of 1..100, and their odd-only and even-only
versions, but a magnitude range is a concept defined by its endpoints, and a
single example cannot suggest one: when only one example is shown, people read
it as an instance of a mathematical/digit property, so prior belief shifts from
range concepts to rules by an inferred amount kappa (kappa = 0 recovers
parity_restricted_concepts). Predicts broad endorsement of the members of the
rules containing a lone example (11 -> 22, 55, 99; 10 -> 60, 90) instead of
belief confined to a tight range around it; multi-example sets are unchanged.
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


def strong_ll_table(membership):
    """Size-principle (strong sampling) log-likelihood of each number."""
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
RULE_MEMBERSHIP = np.concatenate([BASE_RULE, BASE_RULE * ODD, BASE_RULE * EVEN])
_ODD_INT, _ODD_CNT = _dedupe(BASE_INT * ODD)
_EVEN_INT, _EVEN_CNT = _dedupe(BASE_INT * EVEN)
INT_MEMBERSHIP = np.concatenate([BASE_INT, _ODD_INT, _EVEN_INT])
INT_LOG_COUNT = np.log(np.concatenate([np.ones(N_INT), _ODD_CNT, _EVEN_CNT]))
INT_FAMILY = np.concatenate([np.zeros(N_INT), np.ones(len(_ODD_CNT)), np.full(len(_EVEN_CNT), 2)]).astype("int64")
N_INT_ALL = INT_MEMBERSHIP.shape[0]
RULE_STRONG = strong_ll_table(RULE_MEMBERSHIP)
INT_STRONG = strong_ll_table(INT_MEMBERSHIP)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, 3 * N_RULES)))
    interval_log_lik = pm.Data("interval_log_lik", np.zeros((1, N_INT_ALL)))
    single_example = pm.Data("single_example", np.zeros(1))  # 1 if the set has one example
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
    rho = pm.Beta("rho", 1.0, 3.0)
    fam = pt.stack([pt.log1p(-rho), pt.log(rho / 2), pt.log(rho / 2)])
    # Shift of prior log-odds from ranges to rules when one example is shown.
    kappa = pm.Normal("kappa", mu=0.0, sigma=2.0)
    rule_logodds = pt.log(lambda_0) - pt.log1p(-lambda_0) + kappa * single_example  # (sets,)
    log_rule_share = -pt.softplus(-rule_logodds)[:, None]
    log_int_share = -pt.softplus(rule_logodds)[:, None]

    log_rule_prior = (fam[:, None] + pt.log(rule_weights)[None, :]).reshape((-1,))
    log_int_prior = -np.log(N_INT) + fam[INT_FAMILY] + INT_LOG_COUNT
    log_joint = pt.concatenate([
        log_rule_share + log_rule_prior[None, :] + beta * rule_log_lik,
        log_int_share + log_int_prior[None, :] + beta * interval_log_lik,
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
    idx = [np.array(s) - 1 for s in unique]
    return {
        "rule_log_lik": np.stack([RULE_STRONG[:, i].sum(axis=1) for i in idx]),
        "interval_log_lik": np.stack([INT_STRONG[:, i].sum(axis=1) for i in idx]),
        "single_example": np.array([1.0 if len(s) == 1 else 0.0 for s in unique]),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
