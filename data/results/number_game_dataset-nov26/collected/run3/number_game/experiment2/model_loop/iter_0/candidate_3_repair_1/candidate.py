"""Refinement of range_bounded_parity_concepts: range bounds need an example spread.

People do the same tempered-evidence Bayesian concept learning over the 45
weighted rules, all intervals of 1..100, their odd-only / even-only versions
(share rho) and rules restricted to a range [a, b] covering the examples
(share eta). The one change: a range bound is read off the spread of the
examples, so a rule is restricted to a range only when at least two different
examples mark that range out. A single example suggests no bound; it is
generalised through the unbounded rules (and intervals) it belongs to, so
people generalise more broadly from one example (10 -> 60, 70, 80, 90) than
the incumbent allows. No new parameters.
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


def _bounded_groups(examples):
    """Group the covering-range bounded rules (rule r, [a, b], a <= min, b >= max)
    by (rule, size): all members of a group share one likelihood, so the group is
    summarised by its count N, the examples' log-likelihood, and the mean
    membership of each target over the group's ranges. Computed once per set."""
    lo, hi = min(examples), max(examples)
    ex = np.array(examples) - 1
    A, B = np.meshgrid(np.arange(1, lo + 1), np.arange(hi, 101), indexing="ij")
    A, B = A.ravel(), B.ravel()
    covers = ((A[:, None] <= NUMBERS[None, :]) & (B[:, None] >= NUMBERS[None, :])).astype("float64")
    out = []
    for r in range(N_RULES):
        cum = np.concatenate([[0.0], np.cumsum(BASE_RULE[r])])
        size = (cum[B] - cum[A - 1]).astype("int64")
        n_in = BASE_RULE[r, ex]
        for s in np.unique(size):
            sel = size == s
            n = sel.sum()
            ll = np.log(ALPHA * n_in / max(s, 1) + (1 - ALPHA) / 100).sum()
            mem = covers[sel].sum(axis=0) * BASE_RULE[r] / n
            out.append((r, np.log(n), ll, mem))
    return out


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_ll = np.stack([RULE_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    int_ll = np.stack([INT_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    groups = [_bounded_groups(s) for s in unique]
    h = max(len(g) for g in groups)
    n = len(unique)
    b_rule = np.zeros((n, h), dtype="int64")
    b_logn = np.full((n, h), -1e4)
    b_ll = np.zeros((n, h))
    b_mem = np.zeros((n, h, 100))
    for i, g in enumerate(groups):
        if len(set(unique[i])) < 2:
            continue  # the one change: one example marks out no range bound
        for j, (r, ln, ll, mem) in enumerate(g):
            b_rule[i, j], b_logn[i, j], b_ll[i, j], b_mem[i, j] = r, ln, ll, mem
    # Only the (set, target) cells actually asked about are evaluated.
    cell_keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(cell_keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "cell_set": cell_set,
        "cell_flat": (cell_set * 100 + cell_t).astype("int64"),
        "cell_bounded_membership": b_mem[cell_set, :, cell_t],
        "rule_log_lik": rule_ll,
        "interval_log_lik": int_ll,
        "bounded_rule": b_rule,
        "bounded_log_count": b_logn,
        "bounded_log_lik": b_ll,
        "row_cell": np.array([cell_index[c] for c in cell_keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, 3 * N_RULES)))
    interval_log_lik = pm.Data("interval_log_lik", np.zeros((1, N_INT_ALL)))
    bounded_rule = pm.Data("bounded_rule", np.zeros((1, 1), dtype="int64"))
    bounded_log_count = pm.Data("bounded_log_count", np.zeros((1, 1)))
    bounded_log_lik = pm.Data("bounded_log_lik", np.zeros((1, 1)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_flat = pm.Data("cell_flat", np.zeros(1, dtype="int64"))
    cell_bounded_membership = pm.Data("cell_bounded_membership", np.zeros((1, 1)))
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
    rho = pm.Beta("rho", 1.0, 3.0)  # parity-restricted share
    eta = pm.Beta("eta", 1.0, 3.0)  # range-bounded share of the rule prior
    fam = pt.stack([pt.log1p(-rho), pt.log(rho / 2), pt.log(rho / 2)])
    log_w = pt.log(rule_weights)

    # Unbounded rules (plain + parity versions) and intervals, as in parity_restricted_concepts.
    log_rule_prior = (pt.log(lambda_0) + pt.log1p(-eta) + fam[:, None] + log_w[None, :]).reshape((-1,))
    log_int_prior = pt.log1p(-lambda_0) - np.log(N_INT) + fam[INT_FAMILY] + INT_LOG_COUNT
    # Range-bounded rules, grouped by (rule, size): prior lambda_0 * eta * w_r / N_INT per range.
    log_bounded = (pt.log(lambda_0) + pt.log(eta) - np.log(N_INT) + log_w[bounded_rule]
                   + bounded_log_count + beta * bounded_log_lik)
    full = pt.concatenate([
        log_rule_prior[None, :] + beta * rule_log_lik,
        log_int_prior[None, :] + beta * interval_log_lik,
        log_bounded,
    ], axis=1)
    posterior = pt.exp(full - pt.logsumexp(full, axis=1, keepdims=True))
    n_unb = 3 * N_RULES + N_INT_ALL
    unbounded_in = pt.dot(posterior[:, :n_unb], pt.as_tensor(ALL_MEMBERSHIP)).reshape((-1,))[cell_flat]
    bounded_in = (posterior[:, n_unb:][cell_set] * cell_bounded_membership).sum(axis=1)
    in_concept = unbounded_in + bounded_in

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
