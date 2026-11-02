"""Anchored reliability with individually plausible rules.

Refinement of `first_example_anchored_reliability`: Bayesian concept learning
over 45 rules and every interval of 1-100 (size principle; the first example
is trusted to come from the concept with its own inferred reliability a1,
the later examples are pooled into an uncertain Beta(alpha, kappa) belief
about the program's reliability). The one change (from
`family_pooled_uncertain_reliability`): each rule's prior log-weight is drawn
around its family's level with a stated spread RULE_SPREAD, instead of every
rule of a family being equally plausible. "Yes" is the posterior probability
that the concept contains the target.

p(X | h) = [a1 1(x1 in h)/s + (1 - a1)/100] * T(m, k, s), with m = n - 1
later examples of which k fall in h and
T(m, k, s) = sum_j C(k, j) s^-j 100^-(m-j) B(u + j, v + m - j) / B(u, v).
The (m, first, k, s) table is flattened; interval evidence is a dense dot
for sets and a sparse dot for (set, target) cells.
"""

import math

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_M = 3
RULE_SPREAD = 1.75


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
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE_IDX = RULE_MEMBERSHIP.sum(axis=1).astype("int64") - 1
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE_IDX = INT_HI - INT_LO

FAMILIES = ["parity", "primes", "squares and cubes", "powers", "multiples", "ends in", "contains digit"]


def family_of(name):
    if name in ("odd", "even"):
        return "parity"
    if name in ("squares", "cubes"):
        return "squares and cubes"
    for f in ("primes", "powers", "multiples", "ends in", "contains digit"):
        if name.startswith(f):
            return f
    raise ValueError(name)


RULE_FAMILY = np.array([FAMILIES.index(family_of(r)) for r in RULE_NAMES])

# Flattened table index: ((m * 2 + first) * 4 + k) * 100 + (s - 1).
N_TABLE = (MAX_M + 1) * 2 * (MAX_M + 1) * 100


def flat_idx(m, first, k, size_idx):
    return ((m * 2 + first) * (MAX_M + 1) + k) * 100 + size_idx


_M = np.arange(MAX_M + 1, dtype="float64")[:, None, None, None]
_K = np.arange(MAX_M + 1, dtype="float64")[None, :, None, None]
_S = np.arange(1, 101, dtype="float64")[None, None, :, None]
_J = np.arange(MAX_M + 1, dtype="float64")[None, None, None, :]
_VALID = _J <= _K  # k > m cells stay finite (never used), only j > k masked
_LOG_BINOM = np.where(
    _VALID,
    np.vectorize(lambda k, j: math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(max(k - j, 0) + 1))(_K, _J),
    -1e4,
)
_M_MINUS_J = np.maximum(_M - _J, 0.0)
_CONST = _LOG_BINOM - _J * np.log(_S) - _M_MINUS_J * np.log(100.0)
_J_SHAPED = _J + 0.0 * _M_MINUS_J
_LOG_S = np.log(np.arange(1, 101, dtype="float64"))


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    rule_flat_idx = pm.Data("rule_flat_idx", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules)
    interval_count = pm.Data("interval_count", np.zeros((1, N_TABLE)))  # (sets, table)
    cc_data = pm.Data("cc_data", np.zeros(1))
    cc_indices = pm.Data("cc_indices", np.zeros(1, dtype="int32"))
    cc_indptr = pm.Data("cc_indptr", np.array([0, 1], dtype="int32"))
    cs_indices = pm.Data("cs_indices", np.zeros(1, dtype="int32"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    n_sets = interval_count.shape[0]
    n_cells = cs_indices.shape[0]
    cell_count = ps.CSR(cc_data, cc_indices, cc_indptr, pt.stack([n_cells, N_TABLE]))
    cell_set_onehot = ps.CSR(
        pt.ones_like(cs_indices, dtype="float64"), cs_indices, pt.arange(n_cells + 1, dtype="int32"),
        pt.stack([n_cells, n_sets]),
    )

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_level = pm.ZeroSumNormal("family_level", sigma=1.5, shape=len(FAMILIES))
    # The change: each rule has its own plausibility around its family level.
    rule_logw = pm.Normal("rule_logw", mu=family_level[RULE_FAMILY], sigma=RULE_SPREAD, shape=N_RULES)
    rule_weights = pm.Deterministic("rule_weights", pt.special.softmax(rule_logw))
    first_logit = pm.Normal("first_logit", mu=2.2, sigma=1.5)
    a1 = pm.Deterministic("a1", pm.math.sigmoid(first_logit))
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    later_log = pt.logsumexp(_CONST + _betaln(u + _J_SHAPED, v + _M_MINUS_J) - _betaln(u, v), axis=3)  # (m,k,s)
    first_log = pt.stack(
        [
            pt.log1p(-a1) - np.log(100.0) + 0.0 * _LOG_S,
            pt.logaddexp(pt.log(a1) - _LOG_S, pt.log1p(-a1) - np.log(100.0)),
        ]
    )  # (first, s)
    log_table = (first_log[None, :, None, :] + later_log[:, None, :, :]).flatten()
    lik_table = pt.exp(log_table)

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + log_table[rule_flat_idx]
    int_ev = pt.dot(interval_count, lik_table)
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    set_stats = pt.concatenate([rule_post, (interval_post / int_ev)[:, None]], axis=1)
    cell_stats = ps.structured_dot(cell_set_onehot, set_stats)  # (cells, rules + 1)
    cell_int = ps.structured_dot(cell_count, lik_table[:, None])[:, 0]
    in_concept = (cell_stats[:, :N_RULES] * cell_rule_member).sum(axis=1) + cell_stats[:, N_RULES] * cell_int
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_flat = np.zeros((S, N_RULES), dtype="int64")
    int_flat = np.zeros((S, len(INT_LO)), dtype="int64")
    counts = np.zeros((S, N_TABLE))
    for i, s in enumerate(unique):
        ex = np.array(s)
        first, later = ex[0], ex[1:]
        m = len(later)
        rf = RULE_MEMBERSHIP[:, first - 1].astype("int64")
        rk = RULE_MEMBERSHIP[:, later - 1].sum(axis=1).astype("int64")
        rule_flat[i] = flat_idx(m, rf, rk, RULE_SIZE_IDX)
        inf = ((first >= INT_LO) & (first <= INT_HI)).astype("int64")
        ink = ((later[None, :] >= INT_LO[:, None]) & (later[None, :] <= INT_HI[:, None])).sum(axis=1)
        int_flat[i] = flat_idx(m, inf, ink, INT_SIZE_IDX)
        np.add.at(counts[i], int_flat[i], 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    data, indices, indptr = [], [], [0]
    for i, t in cells:
        contains = (INT_LO <= t) & (t <= INT_HI)
        u_idx, cnt = np.unique(int_flat[i][contains], return_counts=True)
        data.extend(cnt.astype("float64"))
        indices.extend(u_idx.astype("int32"))
        indptr.append(len(indices))
    return {
        "rule_flat_idx": rule_flat,
        "interval_count": counts,
        "cc_data": np.array(data, dtype="float64"),
        "cc_indices": np.array(indices, dtype="int32"),
        "cc_indptr": np.array(indptr, dtype="int32"),
        "cs_indices": np.array([i for i, _ in cells], dtype="int32"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
