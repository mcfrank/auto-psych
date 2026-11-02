"""Rule-specific plausibility around family levels, with first-example anchoring.

Refinement of `first_example_anchored_reliability`. People do Bayesian concept
learning over 45 mathematical rules and every magnitude interval of 1-100
(size principle; rules as a group get share lambda_0; each interval gets
(1 - lambda_0) / 100^2), trusting the first example with its own inferred
reliability a1 while pooling only the later examples into an uncertain
(Beta(mean alpha, concentration kappa)) belief about how reliable the program
is on that set. The one change: a rule's prior plausibility is not just its
family's level. Each rule has its own log-weight drawn around its family's
level with a small, fixed spread (0.6), so particular rules (e.g. "powers of
5", "ends in 5") can be a little more or less natural than others of their
kind, while rules the data say little about stay at their family's level.
(The fixed spread avoids the hierarchical funnel that an inferred spread
creates.)

p(X | h) = [a1 1(x1 in h)/s + (1 - a1)/100] * T(m, k, s), where m = n - 1
later examples of which k fall in h, and
T(m, k, s) = sum_j C(k, j) s^-j 100^-(m-j) B(u + j, v + m - j) / B(u, v).
"Yes" is the posterior probability that the concept contains the target.
"""

import math

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_M = 3  # later examples (sets have at most 4)


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
INT_SIZE_IDX = INT_HI - INT_LO  # size - 1

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

# Constant parts of the (m, k, s, j) table for the later examples; m, k, j = 0..3.
_M = np.arange(MAX_M + 1, dtype="float64")[:, None, None, None]
_K = np.arange(MAX_M + 1, dtype="float64")[None, :, None, None]
_S = np.arange(1, 101, dtype="float64")[None, None, :, None]
_J = np.arange(MAX_M + 1, dtype="float64")[None, None, None, :]
_VALID = (_J <= _K) & (_K <= _M)
_LOG_BINOM = np.where(
    _VALID,
    np.vectorize(lambda k, j: math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(max(k - j, 0) + 1))(_K, _J),
    -1e4,
)
_CONST = _LOG_BINOM - _J * np.log(_S) - np.maximum(_M - _J, 0.0) * np.log(100.0)
_M_MINUS_J = np.maximum(_M - _J, 0.0)
_J_SHAPED = _J + 0.0 * _M_MINUS_J
_LOG_S = np.log(np.arange(1, 101, dtype="float64"))


TABLE_SHAPE = (MAX_M + 1, 2, MAX_M + 1, 100)  # (m, first, k, size)
TABLE_SIZE = int(np.prod(TABLE_SHAPE))


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    rule_flat_idx = pm.Data("rule_flat_idx", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules) into flat table
    # (sets, m*first*k*size) interval counts and (cells, ...) counts of the
    # intervals holding the target, as CSR parts; (cells, sets) one-hot of each
    # cell's set as CSR column indices.
    ic_data = pm.Data("ic_data", np.zeros(1))
    ic_indices = pm.Data("ic_indices", np.zeros(1, dtype="int32"))
    ic_indptr = pm.Data("ic_indptr", np.array([0, 1], dtype="int32"))
    cc_data = pm.Data("cc_data", np.zeros(1))
    cc_indices = pm.Data("cc_indices", np.zeros(1, dtype="int32"))
    cc_indptr = pm.Data("cc_indptr", np.array([0, 1], dtype="int32"))
    cs_indices = pm.Data("cs_indices", np.zeros(1, dtype="int32"))
    n_sets = rule_flat_idx.shape[0]
    n_cells = cs_indices.shape[0]
    interval_mat = ps.CSR(ic_data, ic_indices, ic_indptr, pt.stack([n_sets, TABLE_SIZE]))
    cell_mat = ps.CSR(cc_data, cc_indices, cc_indptr, pt.stack([n_cells, TABLE_SIZE]))
    cell_onehot = ps.CSR(
        pt.ones_like(cs_indices, dtype="float64"), cs_indices, pt.arange(n_cells + 1, dtype="int32"),
        pt.stack([n_cells, n_sets]),
    )
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Family-structured rule prior: family levels, rule-specific deviations around them
    # (sum-to-zero levels, so the softmax has no unidentified shift).
    family_level = pm.ZeroSumNormal("family_level", sigma=1.5, shape=len(FAMILIES))
    rule_logw = pm.Normal("rule_logw", mu=family_level[RULE_FAMILY], sigma=0.6, shape=N_RULES)
    rule_weights = pm.Deterministic("rule_weights", pt.special.softmax(rule_logw))
    # Trust in the first (anchoring) example.
    first_logit = pm.Normal("first_logit", mu=2.2, sigma=1.5)
    a1 = pm.Deterministic("a1", pm.math.sigmoid(first_logit))
    # Uncertain reliability for the later examples.
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    later_log = pt.logsumexp(
        _CONST + _betaln(u + _J_SHAPED, v + _M_MINUS_J) - _betaln(u, v), axis=3
    )  # (m, k, s)
    # First example: in the concept (index 1) or not (index 0).
    first_log = pt.stack(
        [
            pt.log1p(-a1) - np.log(100.0) + 0.0 * _LOG_S,
            pt.logaddexp(pt.log(a1) - _LOG_S, pt.log1p(-a1) - np.log(100.0)),
        ]
    )  # (2, s)
    log_table = first_log[None, :, None, :] + later_log[:, None, :, :]  # (m, first, k, s)
    log_flat = log_table.flatten()
    lik_flat = pt.exp(log_flat)

    rule_ll = log_flat[rule_flat_idx]  # (sets, rules)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll

    int_ev = ps.structured_dot(interval_mat, lik_flat[:, None])[:, 0]  # (sets,)
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    # Per (set, target) cell: posterior of each rule and of the intervals as a
    # group, and the share of the intervals' evidence that contains the target.
    post_by_set = pt.concatenate([rule_post, (interval_post / int_ev)[:, None]], axis=1)
    cell_post = ps.structured_dot(cell_onehot, post_by_set)  # (cells, rules + 1)
    cell_int_ev = ps.structured_dot(cell_mat, lik_flat[:, None])[:, 0]
    in_concept = (cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1) + cell_post[:, N_RULES] * cell_int_ev
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _csr(prefix, dense):
    rows, cols = np.nonzero(dense)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=dense.shape[0]))])
    return {
        f"{prefix}_data": dense[rows, cols].astype("float64"),
        f"{prefix}_indices": cols.astype("int32"),
        f"{prefix}_indptr": indptr.astype("int32"),
    }


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_first = np.zeros((S, N_RULES), dtype="int64")
    rule_k = np.zeros((S, N_RULES), dtype="int64")
    set_m_idx = np.zeros(S, dtype="int64")
    int_first = np.zeros((S, len(INT_LO)), dtype="int64")
    int_k = np.zeros((S, len(INT_LO)), dtype="int64")
    counts = np.zeros((S, 2, MAX_M + 1, 100))
    for i, s in enumerate(unique):
        ex = np.array(s)
        first, later = ex[0], ex[1:]
        set_m_idx[i] = len(later)
        rule_first[i] = RULE_MEMBERSHIP[:, first - 1].astype("int64")
        rule_k[i] = RULE_MEMBERSHIP[:, later - 1].sum(axis=1).astype("int64")
        int_first[i] = ((first >= INT_LO) & (first <= INT_HI)).astype("int64")
        int_k[i] = ((later[None, :] >= INT_LO[:, None]) & (later[None, :] <= INT_HI[:, None])).sum(axis=1)
        np.add.at(counts[i], (int_first[i], int_k[i], INT_SIZE_IDX), 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_count = np.zeros((len(cells), 2, MAX_M + 1, 100))
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(
            cell_count[j], (int_first[i][contains], int_k[i][contains], INT_SIZE_IDX[contains]), 1.0
        )
    flat = lambda m, f, k, z: np.ravel_multi_index((m, f, k, z), TABLE_SHAPE)
    rule_flat_idx = flat(set_m_idx[:, None], rule_first, rule_k, RULE_SIZE_IDX[None, :]).astype("int64")
    interval_mat = np.zeros((S, TABLE_SIZE))
    for i in range(S):
        interval_mat[i].reshape(TABLE_SHAPE)[set_m_idx[i]] = counts[i]
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    cell_mat = np.zeros((len(cells), TABLE_SIZE))
    for j in range(len(cells)):
        cell_mat[j].reshape(TABLE_SHAPE)[set_m_idx[cell_set[j]]] = cell_count[j]
    return {
        "rule_flat_idx": rule_flat_idx,
        **_csr("ic", interval_mat),
        **_csr("cc", cell_mat),
        "cs_indices": cell_set.astype("int32"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
