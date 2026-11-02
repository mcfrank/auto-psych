"""Without-replacement anchored concept learning.

Refinement of `first_example_anchored_reliability`: people do Bayesian concept
learning over 45 mathematical rules and every magnitude interval of 1-100
(rules share lambda_0, split equally within kind of rule by one inferred
plausibility per family; each interval gets (1 - lambda_0) / 100^2), the first
example trusted at its own reliability a1, the later examples pooled into an
uncertain Beta(alpha, kappa) reliability belief, strays drawn from 1-100. The
one change: genuine examples are drawn from the concept WITHOUT replacement
(the program never repeats a number in a set), so g genuine examples of a
concept of size s have probability (s-g)!/s! instead of s^-g. No new
parameter. "Yes" is the posterior probability that the concept contains the
target.

With f = 1 if the first example is genuine:
T_f(m, k, s) = sum_j C(k, j) [prod_{i<j} 1/(s-f-i)] 100^-(m-j) B(u+j, v+m-j)/B(u, v)
p(X|h) = a1/s * T_1 + (1-a1)/100 * T_0   if x1 in h,  (1-a1)/100 * T_0 otherwise.
Everything depends only on (m, first_in, k, s): one table per gradient step.
"""

import math

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_M = 3


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

# (m, k, s, j) constants; j > k masked; k > m cells kept finite (never used).
_M = np.arange(MAX_M + 1, dtype="float64")[:, None, None, None]
_K = np.arange(MAX_M + 1, dtype="float64")[None, :, None, None]
_S = np.arange(1, 101, dtype="float64")[None, None, :, None]
_J = np.arange(MAX_M + 1, dtype="float64")[None, None, None, :]
_LOG_BINOM = np.where(
    _J <= _K,
    np.vectorize(lambda k, j: math.lgamma(k + 1) - math.lgamma(j + 1) - math.lgamma(max(k - j, 0) + 1))(_K, _J),
    -1e4,
)
_M_MINUS_J = np.maximum(_M - _J, 0.0) + 0.0 * _S
_J_SHAPED = _J + 0.0 * _M_MINUS_J


def _log_falling(f):
    """log prod_{i<j} 1/(s - f - i): without-replacement draws of j genuine examples."""
    out = np.zeros((1, 1, 100, MAX_M + 1))
    for j in range(1, MAX_M + 1):
        out[..., j] = out[..., j - 1] - np.log(np.maximum(np.arange(1, 101) - f - (j - 1), 1.0))
    return out


_CONST0 = _LOG_BINOM + _log_falling(0) - _M_MINUS_J * np.log(100.0)
_CONST1 = _LOG_BINOM + _log_falling(1) - _M_MINUS_J * np.log(100.0)
_LOG_S = np.log(np.arange(1, 101, dtype="float64"))
N_FULL = (MAX_M + 1) * 2 * (MAX_M + 1) * 100


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


def _csr_data(name, shape_cols):
    data = pm.Data(f"{name}_data", np.zeros(1))
    indices = pm.Data(f"{name}_indices", np.zeros(1, dtype="int32"))
    indptr = pm.Data(f"{name}_indptr", np.array([0, 1], dtype="int32"))
    return ps.CSR(data, indices, indptr, pt.stack([indptr.shape[0] - 1, shape_cols]))


def _matvec(sparse, vec):
    return ps.structured_dot(sparse, vec[:, None])[:, 0]


with pm.Model() as model:
    rule_flat_idx = pm.Data("rule_flat_idx", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules)
    set_m_idx = pm.Data("set_m_idx", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    interval_count = _csr_data("ic", N_FULL)  # (sets, table)
    cell_count = _csr_data("cc", N_FULL)  # (cells, table)
    cell_set_onehot = _csr_data("cs", set_m_idx.shape[0])  # (cells, sets)

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_level = pm.ZeroSumNormal("family_level", sigma=1.5, shape=len(FAMILIES))
    rule_weights = pm.Deterministic("rule_weights", pt.special.softmax(family_level[RULE_FAMILY]))
    first_logit = pm.Normal("first_logit", mu=2.2, sigma=1.5)
    a1 = pm.Deterministic("a1", pm.math.sigmoid(first_logit))
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    beta_term = _betaln(u + _J_SHAPED, v + _M_MINUS_J) - _betaln(u, v)
    later0 = pt.logsumexp(_CONST0 + beta_term, axis=3)  # (m, k, s), first example a stray
    later1 = pt.logsumexp(_CONST1 + beta_term, axis=3)  # (m, k, s), first example genuine
    stray_first = pt.log1p(-a1) - np.log(100.0) + later0
    in_first = pt.logaddexp(pt.log(a1) - _LOG_S[None, None, :] + later1, stray_first)
    log_table = pt.stack([stray_first, in_first], axis=1)  # (m, first, k, s)
    lik_full = pt.exp(log_table).reshape((-1,))

    rule_ll = log_table.reshape((-1,))[rule_flat_idx]  # (sets, rules)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll
    int_ev = _matvec(interval_count, lik_full)
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    cell_rule_post = ps.structured_dot(cell_set_onehot, rule_post)
    cell_int_scale = ps.structured_dot(cell_set_onehot, (interval_post / int_ev)[:, None])[:, 0]
    in_concept = (cell_rule_post * cell_rule_member).sum(axis=1) + cell_int_scale * _matvec(cell_count, lik_full)
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


def _flat(m, first, k, size_idx):
    return ((m * 2 + first) * (MAX_M + 1) + k) * 100 + size_idx


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_flat_idx = np.zeros((S, N_RULES), dtype="int64")
    set_m_idx = np.zeros(S, dtype="int64")
    int_flat = np.zeros((S, len(INT_LO)), dtype="int64")
    counts = np.zeros((S, N_FULL))
    for i, s in enumerate(unique):
        ex = np.array(s)
        first, later = ex[0], ex[1:]
        m = len(later)
        set_m_idx[i] = m
        rf = RULE_MEMBERSHIP[:, first - 1].astype("int64")
        rk = RULE_MEMBERSHIP[:, later - 1].sum(axis=1).astype("int64")
        rule_flat_idx[i] = _flat(m, rf, rk, RULE_SIZE_IDX)
        inf = ((first >= INT_LO) & (first <= INT_HI)).astype("int64")
        ink = ((later[None, :] >= INT_LO[:, None]) & (later[None, :] <= INT_HI[:, None])).sum(axis=1)
        int_flat[i] = _flat(m, inf, ink, INT_SIZE_IDX)
        np.add.at(counts[i], int_flat[i], 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_count = np.zeros((len(cells), N_FULL))
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_count[j], int_flat[i][contains], 1.0)
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    return {
        "rule_flat_idx": rule_flat_idx,
        "set_m_idx": set_m_idx,
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
        **_csr("ic", counts),
        **_csr("cc", cell_count),
        **_csr("cs", np.eye(S)[cell_set]),
    }
