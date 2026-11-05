"""Cumulative explaining away with a guessing lapse.

Refinement of iter2_candidate1 (cumulative explaining away of implied rules).
People infer the program's concept by the same tempered, sample-size-neglecting
Bayesian concept learning (45 weighted rules, size-weighted ranges with their
own evidence strength, odd/even versions, range-bounded rules every example
fits, comparison-based rules discounted for a lone example), and they explain
away broader rules implied by a specific rule all examples share, more with
each further example. The one change, a component taken from
range_evidence_with_guessing_lapse: on some trials (inferred lapse rate) people
do not consult the inferred concept at all and guess, saying yes at an inferred
rate whatever the target is (lapse 0 recovers iter2_candidate1). The yes-floor
for numbers no concept contains then comes from guessing, not from wide
ranges and stray-tolerant broad rules, so the concept itself can be sharper:
more range-confined generalization from pairs and stronger rejection of
same-parity non-matching targets after shared-digit sets.

Implementation note: posterior rows are mapped to (set, target) cells with a
sparse one-hot product and a softmax op (same quantity as posterior[cell_set],
cheaper gradient).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor.sparse as ps

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
RULE_LL = log_lik_table(RULE_MEMBERSHIP)
_ODD_INT, _ODD_CNT = _dedupe(BASE_INT * ODD)
_EVEN_INT, _EVEN_CNT = _dedupe(BASE_INT * EVEN)
INT_MEMBERSHIP = np.concatenate([BASE_INT, _ODD_INT, _EVEN_INT])
INT_COUNT = np.concatenate([np.ones(N_INT), _ODD_CNT, _EVEN_CNT])
INT_FAMILY = np.concatenate([np.zeros(N_INT), np.ones(len(_ODD_CNT)), np.full(len(_EVEN_CNT), 2)]).astype("int64")
INT_SIZE = INT_MEMBERSHIP.sum(axis=1).astype("int64")
INT_LL = log_lik_table(INT_MEMBERSHIP)
# Per family, how many intervals have each size (for normalising the size prior).
FAM_SIZE_COUNT = np.zeros((3, 101))
for _f, _s, _c in zip(INT_FAMILY, INT_SIZE, INT_COUNT):
    FAM_SIZE_COUNT[_f, _s] += _c
LOG_SIZE_GRID = np.log(np.maximum(np.arange(101), 1.0))

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
DIGIT = np.array([1.0 if (nm.startswith("ends in") or nm.startswith("contains")) else 0.0 for nm in RULE_NAMES])
DIGIT3 = np.concatenate([DIGIT, DIGIT, DIGIT])
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0


def _interval_groups(examples):
    """Group intervals (all three families) by (family, size, examples covered):
    members of a group share prior and likelihood, so each is summarised by its
    log count, log-likelihood and mean membership of every target."""
    ex = np.array(examples) - 1
    n_in = INT_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
    ll = INT_LL[:, ex].sum(axis=1)
    key = INT_FAMILY * 100000 + INT_SIZE * 10 + n_in
    out = []
    for k in np.unique(key):
        sel = key == k
        w = INT_COUNT[sel]
        mem = (w[:, None] * INT_MEMBERSHIP[sel]).sum(axis=0) / w.sum()
        out.append((int(INT_FAMILY[sel][0]), float(np.log(max(INT_SIZE[sel][0], 1))),
                    float(np.log(w.sum())), float(ll[sel][0]), mem))
    return out


def _bounded_groups(examples):
    """Covering-range bounded rules grouped by (rule, size), as in the incumbent."""
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
        if n_in.sum() < len(examples):
            continue  # the one change: a bounded rule needs every example to fit
        for s in np.unique(size):
            sel = size == s
            n = sel.sum()
            ll = np.log(ALPHA * n_in / max(s, 1) + (1 - ALPHA) / 100).sum()
            mem = covers[sel].sum(axis=0) * BASE_RULE[r] / n
            out.append((r, np.log(n), ll, mem))
    return out


_SZ = BASE_RULE.sum(axis=1)
STRICT_SUB = ((BASE_RULE @ BASE_RULE.T == _SZ[:, None]) & (_SZ[:, None] < _SZ[None, :])).astype("float64")


def preempted(examples):
    """Per rule: number of examples beyond the first if the rule strictly contains
    another rule every example fits, else 0."""
    fits = (BASE_RULE[:, np.array(examples) - 1].sum(axis=1) == len(examples)).astype("float64")
    w = float(len(examples) - 1)
    return w * (fits @ STRICT_SUB > 0).astype("float64")


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n = len(unique)
    rule_ll = np.stack([RULE_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    igroups = [_interval_groups(s) for s in unique]
    bgroups = [_bounded_groups(s) for s in unique]
    pre_sets = np.stack([preempted(s) for s in unique])
    hi_ = max(len(g) for g in igroups)
    hb = max(max(len(g) for g in bgroups), 1)
    i_fam = np.zeros((n, hi_), dtype="int64")
    i_logsize = np.zeros((n, hi_))
    i_logn = np.full((n, hi_), -1e4)
    i_ll = np.zeros((n, hi_))
    i_mem = np.zeros((n, hi_, 100))
    for i, g in enumerate(igroups):
        for j, (f, ls, ln, ll, mem) in enumerate(g):
            i_fam[i, j], i_logsize[i, j], i_logn[i, j], i_ll[i, j], i_mem[i, j] = f, ls, ln, ll, mem
    b_rule = np.zeros((n, hb), dtype="int64")
    b_logn = np.full((n, hb), -1e4)
    b_ll = np.zeros((n, hb))
    b_mem = np.zeros((n, hb, 100))
    for i, g in enumerate(bgroups):
        for j, (r, ln, ll, mem) in enumerate(g):
            b_rule[i, j], b_logn[i, j], b_ll[i, j], b_mem[i, j] = r, ln, ll, mem
    cell_keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(cell_keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    cell_mem = np.concatenate([RULE_MEMBERSHIP[:, cell_t].T, i_mem[cell_set, :, cell_t],
                               b_mem[cell_set, :, cell_t]], axis=1)
    return {
        "cell_set": cell_set,
        "cell_membership": cell_mem,
        "rule_log_lik": rule_ll,
        "interval_family": i_fam,
        "interval_log_size": i_logsize,
        "interval_log_count": i_logn,
        "interval_log_lik": i_ll,
        "bounded_rule": b_rule,
        "bounded_log_count": b_logn,
        "bounded_log_lik": b_ll,
        "rule_preempted": np.concatenate([pre_sets] * 3, axis=1),
        "bounded_preempted": np.take_along_axis(pre_sets, b_rule, axis=1) * (b_logn > -1e3),
        "single_example": np.array([1.0 if len(s) == 1 else 0.0 for s in unique]),
        "centered_log_n": np.log(np.array([len(s) for s in unique], dtype="float64") / 2.0),
        "row_cell": np.array([cell_index[c] for c in cell_keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_membership = pm.Data("cell_membership", np.zeros((1, 3)))
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, 3 * N_RULES)))
    interval_family = pm.Data("interval_family", np.zeros((1, 1), dtype="int64"))
    interval_log_size = pm.Data("interval_log_size", np.zeros((1, 1)))
    interval_log_count = pm.Data("interval_log_count", np.zeros((1, 1)))
    interval_log_lik = pm.Data("interval_log_lik", np.zeros((1, 1)))
    bounded_rule = pm.Data("bounded_rule", np.zeros((1, 1), dtype="int64"))
    bounded_log_count = pm.Data("bounded_log_count", np.zeros((1, 1)))
    bounded_log_lik = pm.Data("bounded_log_lik", np.zeros((1, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    centered_log_n = pm.Data("centered_log_n", np.zeros(1))
    single_example = pm.Data("single_example", np.zeros(1))
    rule_preempted = pm.Data("rule_preempted", np.zeros((1, 3 * N_RULES)))
    bounded_preempted = pm.Data("bounded_preempted", np.zeros((1, 1)))
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
    neglect = pm.Normal("neglect", 0.0, 0.5)
    beta = pt.exp(log_beta - neglect * centered_log_n)[:, None]
    # The one change: magnitude ranges get their own (shifted) evidence strength.
    range_evidence_shift = pm.Normal("range_evidence_shift", mu=0.0, sigma=1.0)
    beta_range = pt.exp(log_beta + range_evidence_shift - neglect * centered_log_n)[:, None]
    rho = pm.Beta("rho", 1.0, 3.0)  # parity-restricted share
    eta = pm.Beta("eta", 1.0, 3.0)  # range-bounded share of the rule prior
    # Interval prior proportional to size ** size_power, normalised per family.
    size_power = pm.Normal("size_power", mu=0.0, sigma=1.0)
    fam = pt.stack([pt.log1p(-rho), pt.log(rho / 2), pt.log(rho / 2)])
    log_z = pt.logsumexp(size_power * LOG_SIZE_GRID[None, :] + np.log(FAM_SIZE_COUNT + 1e-300), axis=1)
    log_w = pt.log(rule_weights)

    log_rule_prior = (pt.log(lambda_0) + pt.log1p(-eta) + fam[:, None] + log_w[None, :]).reshape((-1,))
    log_int = (pt.log1p(-lambda_0) + fam[interval_family] - log_z[interval_family]
               + size_power * interval_log_size + interval_log_count + beta_range * interval_log_lik)
    log_bounded = (pt.log(lambda_0) + pt.log(eta) - np.log(N_INT) + log_w[bounded_rule]
                   + bounded_log_count + beta * bounded_log_lik)
    comparison_neglect = pm.HalfNormal("comparison_neglect", sigma=3.0)
    pen = comparison_neglect * single_example[:, None]
    # Bounded rules (defined by the examples' span) are also discounted for one example.
    log_bounded = log_bounded - pen * pt.as_tensor_variable(DIGIT)[bounded_rule] - pen
    # The one change: implied (superset) rules are explained away, more with each example.
    preemption = pm.HalfNormal("preemption", sigma=3.0)
    log_bounded = log_bounded - preemption * bounded_preempted
    full = pt.concatenate([log_rule_prior[None, :] - pen * DIGIT3[None, :] - preemption * rule_preempted + beta * rule_log_lik, log_int, log_bounded], axis=1)
    n_cells = cell_set.shape[0]
    onehot = ps.CSR(pt.ones((n_cells,)), cell_set, pt.arange(n_cells + 1), pt.stack([n_cells, rule_log_lik.shape[0]]))
    posterior = pt.special.softmax(full, axis=1)
    in_concept = (ps.structured_dot(onehot, posterior) * cell_membership).sum(axis=1)

    # The one change: on a lapse trial people guess instead of consulting the concept.
    lapse = pm.Beta("lapse", 1.0, 4.0)
    guess = pm.Beta("guess", 2.0, 2.0)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * in_concept[row_cell] + lapse * guess,
                                              1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
