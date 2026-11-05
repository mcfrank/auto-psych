"""Refinement of parity_restricted_concepts: range-bounded rules.

People do the same tempered-evidence Bayesian concept learning over the 45
weighted rules, all intervals of 1..100, and their odd-only / even-only
versions (share rho). The one change, a component taken from
bounded_range_rule_learning: people can also restrict a rule to a range
[a, b] that covers the examples ("multiples of 3 between 20 and 100"). These
bounded rules take a share eta of the rule prior (eta = 0 recovers
parity_restricted_concepts). Because a bounded rule is smaller, the size
principle favours it, so rule members outside the examples' min-max range
get less endorsement than rule members inside it.
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
RULE_LL = log_lik_table(RULE_MEMBERSHIP)
INT_LL = log_lik_table(INT_MEMBERSHIP)
ALL_MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INT_MEMBERSHIP])

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0


def _bounded_tables(examples):
    """Log-likelihood of the examples under each (rule, [a, b]) with a <= min, b >= max."""
    lo, hi = min(examples), max(examples)
    ex = np.array(examples) - 1
    rm_ex = BASE_RULE[:, ex]
    cum = np.concatenate([np.zeros((N_RULES, 1)), np.cumsum(BASE_RULE, axis=1)], axis=1)
    a = np.arange(1, lo + 1)
    b = np.arange(hi, 101)
    size = cum[:, b][:, None, :] - cum[:, a - 1][:, :, None]  # (R, A, B)
    p = ALPHA * rm_ex[:, None, None, :] / np.maximum(size, 1.0)[..., None] + (1 - ALPHA) / 100
    return np.log(p).sum(axis=3), lo, hi


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_ll = np.stack([RULE_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    int_ll = np.stack([INT_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    ll_all, set_all, rule_all, a_all, b_all, set_max = [], [], [], [], [], []
    for s, ex in enumerate(unique):
        ll, lo, hi = _bounded_tables(ex)
        r, i, j = np.meshgrid(np.arange(N_RULES), np.arange(lo), np.arange(101 - hi), indexing="ij")
        ll_all.append(ll.ravel())
        set_all.append(np.full(ll.size, s))
        rule_all.append(r.ravel())
        a_all.append(i.ravel())
        b_all.append(hi - 1 + j.ravel())
        set_max.append(ll.max())
    set_h = np.concatenate(set_all)
    sr = set_h * N_RULES + np.concatenate(rule_all)
    return {
        "rule_log_lik": rule_ll,
        "interval_log_lik": int_ll,
        "hyp_log_lik": np.concatenate(ll_all),
        "hyp_set": set_h.astype("int64"),
        "hyp_rule": np.concatenate(rule_all).astype("int64"),
        "hyp_sr": sr.astype("int64"),
        "hyp_sra": (sr * 100 + np.concatenate(a_all)).astype("int64"),
        "hyp_srb": (sr * 100 + np.concatenate(b_all)).astype("int64"),
        "set_max_log_lik": np.array(set_max),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, 3 * N_RULES)))
    interval_log_lik = pm.Data("interval_log_lik", np.zeros((1, N_INT_ALL)))
    hyp_log_lik = pm.Data("hyp_log_lik", np.zeros(1))
    hyp_set = pm.Data("hyp_set", np.zeros(1, dtype="int64"))
    hyp_rule = pm.Data("hyp_rule", np.zeros(1, dtype="int64"))
    hyp_sr = pm.Data("hyp_sr", np.zeros(1, dtype="int64"))
    hyp_sra = pm.Data("hyp_sra", np.zeros(1, dtype="int64"))
    hyp_srb = pm.Data("hyp_srb", np.zeros(1, dtype="int64"))
    set_max_log_lik = pm.Data("set_max_log_lik", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    n_sets = set_max_log_lik.shape[0]

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
    rho = pm.Beta("rho", 1.0, 3.0)  # parity-restricted share
    eta = pm.Beta("eta", 1.0, 3.0)  # range-bounded share of the rule prior (the one change)
    fam = pt.stack([pt.log1p(-rho), pt.log(rho / 2), pt.log(rho / 2)])

    # Unbounded rules (plain + parity versions) and intervals, as in parity_restricted_concepts.
    log_rule_prior = (pt.log(lambda_0) + pt.log1p(-eta) + fam[:, None] + pt.log(rule_weights)[None, :]).reshape((-1,))
    log_int_prior = pt.log1p(-lambda_0) - np.log(N_INT) + fam[INT_FAMILY] + INT_LOG_COUNT
    log_joint = pt.concatenate([
        log_rule_prior[None, :] + beta * rule_log_lik,
        log_int_prior[None, :] + beta * interval_log_lik,
    ], axis=1)

    # Range-bounded rules: prior lambda_0 * eta * w_r / N_INT for each covering range [a, b].
    log_b_prior = pt.log(lambda_0) + pt.log(eta) + pt.log(rule_weights) - np.log(N_INT)
    shift = beta * set_max_log_lik + pt.max(log_b_prior)
    w = pt.exp(log_b_prior[hyp_rule] + beta * hyp_log_lik - shift[hyp_set])
    Zb = pt.inc_subtensor(pt.zeros(n_sets)[hyp_set], w)
    log_Zb = pt.log(Zb) + shift
    w_sr = pt.inc_subtensor(pt.zeros(n_sets * N_RULES)[hyp_sr], w).reshape((n_sets, N_RULES))
    w_a = pt.inc_subtensor(pt.zeros(n_sets * N_RULES * 100)[hyp_sra], w).reshape((n_sets, N_RULES, 100))
    w_b = pt.inc_subtensor(pt.zeros(n_sets * N_RULES * 100)[hyp_srb], w).reshape((n_sets, N_RULES, 100))
    a_le_t = pt.cumsum(w_a, axis=2)
    b_ge_t = pt.cumsum(w_b[:, :, ::-1], axis=2)[:, :, ::-1]
    in_range = a_le_t + b_ge_t - w_sr[:, :, None]
    bounded_in = (in_range * pt.as_tensor(BASE_RULE)[None, :, :]).sum(axis=1) / Zb[:, None]

    full = pt.concatenate([log_joint, log_Zb[:, None]], axis=1)
    posterior = pt.exp(full - pt.logsumexp(full, axis=1, keepdims=True))
    in_concept = pt.dot(posterior[:, :-1], pt.as_tensor(ALL_MEMBERSHIP)) + posterior[:, -1][:, None] * bounded_in

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
