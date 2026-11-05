"""Refinement of comparison_dependent_concepts: an inferred belief in stray examples.

People do the same tempered, sample-size-neglecting Bayesian concept learning as
comparison_dependent_concepts (45 weighted rules, size-weighted intervals,
odd/even versions, range-bounded rules; digit and bounded rules discounted for a
lone example). The one change: the share of examples people believe to be
random strays from 1-100 is not fixed at 10% but inferred (stray_rate). If
people think strays are rare, any concept that an example contradicts loses
almost all belief, which then moves to the ranges that cover every example. So
in-range targets are endorsed more for sets that no rule fully fits and for
pairs of examples.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# Empty concepts (e.g. odd powers of 2) get an enormous size so a "hit" is impossible.
RULE_LOG_SIZE = np.where(RULE_SIZE > 0, np.log(np.maximum(RULE_SIZE, 1)), 50.0)
_ODD_INT, _ODD_CNT = _dedupe(BASE_INT * ODD)
_EVEN_INT, _EVEN_CNT = _dedupe(BASE_INT * EVEN)
INT_MEMBERSHIP = np.concatenate([BASE_INT, _ODD_INT, _EVEN_INT])
INT_COUNT = np.concatenate([np.ones(N_INT), _ODD_CNT, _EVEN_CNT])
INT_FAMILY = np.concatenate([np.zeros(N_INT), np.ones(len(_ODD_CNT)), np.full(len(_EVEN_CNT), 2)]).astype("int64")
INT_SIZE = INT_MEMBERSHIP.sum(axis=1).astype("int64")
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
    key = INT_FAMILY * 100000 + INT_SIZE * 10 + n_in
    out = []
    for k in np.unique(key):
        sel = key == k
        w = INT_COUNT[sel]
        mem = (w[:, None] * INT_MEMBERSHIP[sel]).sum(axis=0) / w.sum()
        out.append((int(INT_FAMILY[sel][0]), float(np.log(max(INT_SIZE[sel][0], 1))),
                    float(np.log(w.sum())), float(n_in[sel][0]), mem))
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
        if n_in.sum() < len(examples) - 1:
            continue  # two or more contradicting examples: negligible belief, skipped for speed
        for s in np.unique(size):
            sel = size == s
            n = sel.sum()
            mem = covers[sel].sum(axis=0) * BASE_RULE[r] / n
            out.append((r, np.log(n), (float(n_in.sum()), float(np.log(max(s, 1)))), mem))
    return out


BLOCK_SIZES = (1, 2, 3, 4)  # sets are processed in blocks by how many examples they show


def _block(unique, cells):
    """Arrays for one block of example sets (all with the same number of examples);
    each block is padded only to its own widths, which keeps the fit fast."""
    n = len(unique)
    rule_hits = np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    igroups = [_interval_groups(s) for s in unique]
    bgroups = [_bounded_groups(s) for s in unique]
    hi_ = max(len(g) for g in igroups)
    hb = max(max(len(g) for g in bgroups), 1)
    i_fam = np.zeros((n, hi_), dtype="int64")
    i_logsize = np.zeros((n, hi_))
    i_logn = np.full((n, hi_), -1e4)
    i_hits = np.zeros((n, hi_))
    i_mem = np.zeros((n, hi_, 100))
    for i, g in enumerate(igroups):
        for j, (f, ls, ln, h, mem) in enumerate(g):
            i_fam[i, j], i_logsize[i, j], i_logn[i, j], i_hits[i, j], i_mem[i, j] = f, ls, ln, h, mem
    b_rule = np.zeros((n, hb), dtype="int64")
    b_logn = np.full((n, hb), -1e4)
    b_hits = np.zeros((n, hb))
    b_logsize = np.zeros((n, hb))
    b_mem = np.zeros((n, hb, 100))
    for i, g in enumerate(bgroups):
        for j, (r, ln, (h, ls), mem) in enumerate(g):
            b_rule[i, j], b_logn[i, j], b_hits[i, j], b_logsize[i, j], b_mem[i, j] = r, ln, h, ls, mem
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    cell_mem = np.concatenate([RULE_MEMBERSHIP[:, cell_t].T, i_mem[cell_set, :, cell_t],
                               b_mem[cell_set, :, cell_t]], axis=1)
    sizes = np.array([len(s) for s in unique], dtype="float64")
    return {
        "cell_set": cell_set, "cell_membership": cell_mem, "rule_hits": rule_hits,
        "interval_family": i_fam, "interval_log_size": i_logsize, "interval_log_count": i_logn,
        "interval_hits": i_hits, "bounded_rule": b_rule, "bounded_log_count": b_logn,
        "bounded_hits": b_hits, "bounded_log_size": b_logsize,
        "single_example": (sizes == 1).astype("float64"), "n_examples": sizes,
        "centered_log_n": np.log(sizes / 2.0),
    }


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    cell_keys = [(s, int(r["target"])) for s, r in zip(sets, rows)]
    out, order = {}, {}
    offset = 0
    for k_ex in BLOCK_SIZES:
        suffix = f"_n{k_ex}"
        keep = lambda s, k_ex=k_ex: len(s) == k_ex
        unique = list(dict.fromkeys(s for s in sets if keep(s)))
        index = {s: i for i, s in enumerate(unique)}
        cells = list(dict.fromkeys((index[s], t) for s, t in cell_keys if keep(s)))
        if not unique:  # no set of this size: a dummy cell that no row refers to
            unique, cells = [tuple(range(1, k_ex + 1))], [(0, 1)]
        for j, (i, t) in enumerate(cells):
            order[(unique[i], t)] = offset + j
        offset += len(cells)
        for key, val in _block(unique, cells).items():
            out[key + suffix] = val
    out["row_cell"] = np.array([order[c] for c in cell_keys], dtype="int64")
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out


with pm.Model() as model:
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
    neglect = pm.Normal("neglect", 0.0, 0.5)
    rho = pm.Beta("rho", 1.0, 3.0)  # parity-restricted share
    eta = pm.Beta("eta", 1.0, 3.0)  # range-bounded share of the rule prior
    # Interval prior proportional to size ** size_power, normalised per family.
    size_power = pm.Normal("size_power", mu=0.0, sigma=1.0)
    comparison_neglect = pm.HalfNormal("comparison_neglect", sigma=3.0)
    # The one change: the believed stray rate is inferred rather than fixed at 0.1.
    logit_stray = pm.Normal("logit_stray", mu=-2.2, sigma=1.5)
    stray = pm.Deterministic("stray_rate", pm.math.sigmoid(logit_stray))
    log_miss = pt.log(stray / 100.0)

    fam = pt.stack([pt.log1p(-rho), pt.log(rho / 2), pt.log(rho / 2)])
    log_z = pt.logsumexp(size_power * LOG_SIZE_GRID[None, :] + np.log(FAM_SIZE_COUNT + 1e-300), axis=1)
    log_w = pt.log(rule_weights)
    log_rule_prior = (pt.log(lambda_0) + pt.log1p(-eta) + fam[:, None] + log_w[None, :]).reshape((-1,))

    def block_in_concept(sfx):
        d = {k: pm.Data(k + sfx, v) for k, v in (
            ("cell_set", np.zeros(1, dtype="int64")), ("cell_membership", np.zeros((1, 3))),
            ("rule_hits", np.zeros((1, 3 * N_RULES))),
            ("interval_family", np.zeros((1, 1), dtype="int64")), ("interval_log_size", np.zeros((1, 1))),
            ("interval_log_count", np.zeros((1, 1))), ("interval_hits", np.zeros((1, 1))),
            ("bounded_rule", np.zeros((1, 1), dtype="int64")), ("bounded_log_count", np.zeros((1, 1))),
            ("bounded_hits", np.zeros((1, 1))), ("bounded_log_size", np.zeros((1, 1))),
            ("single_example", np.zeros(1)), ("n_examples", np.ones(1)), ("centered_log_n", np.zeros(1)))}
        beta = pt.exp(log_beta - neglect * d["centered_log_n"])[:, None]
        n_ex = d["n_examples"][:, None]

        def ex_log_lik(hits, log_size):
            # Each example is a stray from 1-100 (prob. stray_rate) or drawn from the concept.
            log_hit = pt.logaddexp(pt.log1p(-stray) - log_size, log_miss)
            return hits * log_hit + (n_ex - hits) * log_miss

        fi = d["interval_family"]
        log_int = (pt.log1p(-lambda_0) + fam[fi] - log_z[fi] + size_power * d["interval_log_size"]
                   + d["interval_log_count"] + beta * ex_log_lik(d["interval_hits"], d["interval_log_size"]))
        br = d["bounded_rule"]
        pen = comparison_neglect * d["single_example"][:, None]
        # Bounded rules (defined by the examples' span) are discounted for one example, as digit rules.
        log_bounded = (pt.log(lambda_0) + pt.log(eta) - np.log(N_INT) + log_w[br] + d["bounded_log_count"]
                       + beta * ex_log_lik(d["bounded_hits"], d["bounded_log_size"])
                       - pen * pt.as_tensor_variable(DIGIT)[br] - pen)
        log_rules = (log_rule_prior[None, :] - pen * DIGIT3[None, :]
                     + beta * ex_log_lik(d["rule_hits"], RULE_LOG_SIZE[None, :]))
        full = pt.concatenate([log_rules, log_int, log_bounded], axis=1)
        posterior = pt.exp(full - pt.logsumexp(full, axis=1, keepdims=True))
        return (posterior[d["cell_set"]] * d["cell_membership"]).sum(axis=1)

    in_concept = pt.concatenate([block_in_concept(f"_n{k}") for k in BLOCK_SIZES])
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
