"""Range-restricted rule concepts.

People do Bayesian concept learning over 45 mathematical rules and every
magnitude interval of 1-100 (size principle; rules grouped by family, each
family with one inferred plausibility shared equally by its rules; the first
example trusted at its own inferred reliability a1, the later ones pooled into
an uncertain Beta(alpha, kappa) belief about how often an example is a stray
draw from 1-100) -- and, in addition, they entertain *range-restricted rules*:
a rule's members within a magnitude range ("multiples of 4 between 52 and 88",
"even numbers below 30"). The one change is this enlarged concept space. A
priori the restricted rules hold a stated share lambda_c of the prior
(Beta(2, 38), about 5%), split among rules by the same family-based rule
weights and, within a rule, equally over every interval that cuts it to a
proper, non-empty subset. "Yes" is the posterior probability that the concept
contains the target.

Each concept's likelihood depends only on (m, first_in, k, size), so the
incumbent's table is reused; prepare_observed counts, per set and rule, the
restricting intervals by (first_in, k, size), and per (set, target) cell the
same counts over intervals containing the target for rules containing it.
"""

import math

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt


import numpy as np
import pymc as pm
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


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


MAX_RS = 50  # largest rule size
RULE_CUM = np.concatenate([np.zeros((N_RULES, 1)), np.cumsum(RULE_MEMBERSHIP, axis=1)], axis=1)
# Size of rule r cut to interval i, (rules, intervals); proper non-empty cuts only.
CUT_SIZE = (RULE_CUM[:, INT_HI] - RULE_CUM[:, INT_LO - 1]).astype("int64")
CUT_OK = (CUT_SIZE >= 1) & (CUT_SIZE < RULE_SIZE_IDX[:, None] + 1)
N_CUTS = CUT_OK.sum(axis=1).astype("float64")
MAX_PAIRS = int(RULE_MEMBERSHIP.sum(axis=0).max())


def _csr_data(name, shape_cols):
    data = pm.Data(f"{name}_data", np.zeros(1))
    indices = pm.Data(f"{name}_indices", np.zeros(1, dtype="int32"))
    indptr = pm.Data(f"{name}_indptr", np.array([0, 1], dtype="int32"))
    return ps.CSR(data, indices, indptr, pt.stack([indptr.shape[0] - 1, shape_cols]))


def _matvec(sparse, vec):
    return ps.structured_dot(sparse, vec[:, None])[:, 0]


N_FULL = (MAX_M + 1) * 2 * (MAX_M + 1) * 100  # (m, first, k, size) table cells
N_CUT = (MAX_M + 1) * 2 * (MAX_M + 1) * MAX_RS  # same, sizes up to 50


with pm.Model() as model:
    rule_first = pm.Data("rule_first", np.zeros((1, N_RULES), dtype="int64"))
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES), dtype="int64"))
    set_m_idx = pm.Data("set_m_idx", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    pair_valid = pm.Data("pair_valid", np.zeros((1, MAX_PAIRS)))  # (cells, slots)
    pair_rule = pm.Data("pair_rule", np.zeros((1, MAX_PAIRS), dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    # Sparse count tables over the flattened (m, first, k, size) likelihood table.
    interval_count = _csr_data("ic", N_FULL)  # (sets, .)
    cell_count = _csr_data("cc", N_FULL)  # (cells, .)
    cut_count = _csr_data("cu", N_CUT)  # (sets * rules, .)
    pair_count = _csr_data("pc", N_CUT)  # (cells * slots, .)
    cell_set_onehot = _csr_data("cs", set_m_idx.shape[0])  # (cells, sets)
    n_sets = set_m_idx.shape[0]
    n_cells = pair_valid.shape[0]

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    lambda_c = pm.Beta("lambda_c", 2, 38)
    family_level = pm.ZeroSumNormal("family_level", sigma=1.5, shape=len(FAMILIES))
    rule_weights = pm.Deterministic("rule_weights", pt.special.softmax(family_level[RULE_FAMILY]))
    first_logit = pm.Normal("first_logit", mu=2.2, sigma=1.5)
    a1 = pm.Deterministic("a1", pm.math.sigmoid(first_logit))
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    later_log = pt.logsumexp(_CONST + _betaln(u + _J_SHAPED, v + _M_MINUS_J) - _betaln(u, v), axis=3)
    first_log = pt.stack(
        [
            pt.log1p(-a1) - np.log(100.0) + 0.0 * _LOG_S,
            pt.logaddexp(pt.log(a1) - _LOG_S, pt.log1p(-a1) - np.log(100.0)),
        ]
    )
    log_table = first_log[None, :, None, :] + later_log[:, None, :, :]  # (m, first, k, s)
    lik_table = pt.exp(log_table)
    lik_full = lik_table.reshape((-1,))
    lik_cut = lik_table[:, :, :, :MAX_RS].reshape((-1,))

    log_w = pt.log(rule_weights)
    rule_ll = log_table[set_m_idx[:, None], rule_first, rule_k, RULE_SIZE_IDX[None, :]]
    log_rules = pt.log1p(-lambda_c) + pt.log(lambda_0) + log_w[None, :] + rule_ll

    int_ev = _matvec(interval_count, lik_full)  # (sets,)
    log_intervals = pt.log1p(-lambda_c) + pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    # Range-restricted rules: evidence summed over each rule's proper cuts.
    cut_ev = _matvec(cut_count, lik_cut).reshape((n_sets, N_RULES))
    log_cut_prior = pt.log(lambda_c) + log_w - np.log(N_CUTS)  # (rules,)
    log_cuts = log_cut_prior[None, :] + pt.log(cut_ev + 1e-300)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None], log_cuts], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    def per_cell(x):  # gather a per-set quantity to cells
        return ps.structured_dot(cell_set_onehot, x if x.ndim == 2 else x[:, None])

    cell_int_share = per_cell(interval_post / int_ev)[:, 0] * _matvec(cell_count, lik_full)
    pair_ev = _matvec(pair_count, lik_cut).reshape((n_cells, MAX_PAIRS))
    cut_in = (
        pair_valid * pt.exp(log_cut_prior[pair_rule] - per_cell(log_norm)) * pair_ev
    ).sum(axis=1)
    in_concept = (per_cell(rule_post) * cell_rule_member).sum(axis=1) + cell_int_share + cut_in
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


def _with_m(m, flat):
    """Place a (first, k, size) count vector in block m of the (m, ...) table."""
    out = np.zeros((MAX_M + 1, flat.size))
    out[m] = flat
    return out.reshape(-1)


def _flat_counts(first, k, size, ok):
    out = np.zeros(2 * (MAX_M + 1) * MAX_RS)
    idx = (first[ok] * (MAX_M + 1) + k[ok]) * MAX_RS + (size[ok] - 1)
    np.add.at(out, idx, 1.0)
    return out


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    F = 2 * (MAX_M + 1) * MAX_RS
    rule_first = np.zeros((S, N_RULES), dtype="int64")
    rule_k = np.zeros((S, N_RULES), dtype="int64")
    set_m_idx = np.zeros(S, dtype="int64")
    int_first = np.zeros((S, len(INT_LO)), dtype="int64")
    int_k = np.zeros((S, len(INT_LO)), dtype="int64")
    counts = np.zeros((S, 2, MAX_M + 1, 100))
    cut_first, cut_k = [], []
    cut_count = np.zeros((S, N_RULES, F))
    for i, s in enumerate(unique):
        ex = np.array(s)
        first, later = ex[0], ex[1:]
        set_m_idx[i] = len(later)
        rule_first[i] = RULE_MEMBERSHIP[:, first - 1].astype("int64")
        rule_k[i] = RULE_MEMBERSHIP[:, later - 1].sum(axis=1).astype("int64")
        in_first = (first >= INT_LO) & (first <= INT_HI)
        in_later = (later[None, :] >= INT_LO[:, None]) & (later[None, :] <= INT_HI[:, None])  # (ints, m)
        int_first[i] = in_first.astype("int64")
        int_k[i] = in_later.sum(axis=1)
        np.add.at(counts[i], (int_first[i], int_k[i], INT_SIZE_IDX), 1.0)
        cf = (RULE_MEMBERSHIP[:, first - 1][:, None] * in_first[None, :]).astype("int64")  # (rules, ints)
        ck = (RULE_MEMBERSHIP[:, later - 1] @ in_later.T.astype("float64")).astype("int64")  # (rules, ints)
        cut_first.append(cf)
        cut_k.append(ck)
        for r in range(N_RULES):
            cut_count[i, r] = _flat_counts(cf[r], ck[r], CUT_SIZE[r], CUT_OK[r])
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    C = len(cells)
    cell_count = np.zeros((C, 2, MAX_M + 1, 100))
    pair_count = np.zeros((C, MAX_PAIRS, F))
    pair_rule = np.zeros((C, MAX_PAIRS), dtype="int64")
    pair_valid = np.zeros((C, MAX_PAIRS))
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_count[j], (int_first[i][contains], int_k[i][contains], INT_SIZE_IDX[contains]), 1.0)
        for slot, r in enumerate(np.flatnonzero(RULE_MEMBERSHIP[:, t - 1])):
            pair_rule[j, slot] = r
            pair_valid[j, slot] = 1.0
            pair_count[j, slot] = _flat_counts(cut_first[i][r], cut_k[i][r], CUT_SIZE[r], CUT_OK[r] & contains)
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    cell_m = set_m_idx[cell_set]
    return {
        "rule_first": rule_first,
        "rule_k": rule_k,
        "set_m_idx": set_m_idx,
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "pair_valid": pair_valid,
        "pair_rule": pair_rule,
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
        **_csr("ic", np.stack([_with_m(set_m_idx[i], counts[i].reshape(-1)) for i in range(S)])),
        **_csr("cc", np.stack([_with_m(cell_m[j], cell_count[j].reshape(-1)) for j in range(C)])),
        **_csr("cu", np.stack([_with_m(set_m_idx[i], cut_count[i, r]) for i in range(S) for r in range(N_RULES)])),
        **_csr("pc", np.stack([_with_m(cell_m[j], pair_count[j, q]) for j in range(C) for q in range(MAX_PAIRS)])),
        **_csr("cs", np.eye(S)[cell_set]),
    }
