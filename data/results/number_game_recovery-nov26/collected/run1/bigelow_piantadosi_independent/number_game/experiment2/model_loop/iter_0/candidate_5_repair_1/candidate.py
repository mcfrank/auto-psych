"""Parity-restricted compressed-line ranges with an Erlang prior on range span.

Refinement of `parity_restricted_compressed_intervals`: people do the same
concept learning over rules, plain ranges and parity-restricted ranges sampled
on a compressed number line, but they expect a range to span a moderate stretch
of the number line. The one change (grafted from
`erlang_width_compressed_intervals`): within each range family the prior over
ranges follows an Erlang density on the span s = b - a + 1,
P(range) proportional to s exp(-s / sigma) (sigma fitted), replacing the flat
prior, so tiny ranges hugging one or two examples are a priori unlikely.

Parent model: the same 45 rules, prior
lambda_0 split equally over rules, tempered size principle (beta), fitted
exception rate eps, and range members sampled along a compressed number line,
P(x | range) = x^-gamma / sum_{k in range} k^-gamma. The one change: besides
plain ranges [a, b], the magnitude concepts include parity-restricted ranges
("even numbers in [a, b]", "odd numbers in [a, b]"), which take a fitted share
rho of the prior mass given to ranges. When the examples share a parity, near
same-parity numbers are endorsed while immediate (parity-flipping) neighbours
and the neighbours of an off-pattern example are endorsed less.
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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# Range families: 0 = all numbers, 1 = even numbers, 2 = odd numbers.
FAMILY_MASK = np.array([np.ones(100), NUMBERS % 2 == 0, NUMBERS % 2 == 1], dtype="float64")
_A = np.array([a for a in NUMBERS for b in NUMBERS if b >= a])
_B = np.array([b for a in NUMBERS for b in NUMBERS if b >= a])
# Each distinct parity range is listed once, by endpoints of its own parity.
_valid = [np.arange(len(_A))] + [
    np.nonzero((FAMILY_MASK[f, _A - 1] > 0) & (FAMILY_MASK[f, _B - 1] > 0))[0] for f in (1, 2)
]
CON_A = np.concatenate([_A[v] for v in _valid])
CON_B = np.concatenate([_B[v] for v in _valid])
CON_F = np.concatenate([np.full(len(v), f) for f, v in enumerate(_valid)])
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
CON_S = CON_B - CON_A  # span - 1
SIZES = np.arange(1, 101, dtype="float64")
FAM_SIZE_COUNT = np.stack([np.bincount(CON_S[CON_F == f], minlength=100) for f in range(3)])
LOG_FAM_SIZE_COUNT = np.where(FAM_SIZE_COUNT > 0, np.log(np.maximum(FAM_SIZE_COUNT, 1)), -1e4)


# Outward cumulative tables, flattened: T(f, k, a, b) sums the weight of every
# range of family f holding k examples whose lower end is <= a and upper end >= b.
VALID = np.zeros((3, 100, 100))
VALID[CON_F, CON_A - 1, CON_B - 1] = 1.0
SPAN_GRID = np.clip(NUMBERS[None, :] - NUMBERS[:, None], 0, 99)  # (a, b): span - 1


def table_index(f, k, a, b):
    """Flat position of T(f, k, a, b); a in 0..100 (0 = before 1), b in 1..101 (101 = past 100)."""
    return ((f * 4 + (k - 1)) * 101 + a) * 101 + (b - 1)


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    # Ranges holding >= 1 example come in blocks (same examples inside) that are
    # rectangles of lower x upper ends; each block is read off T by its 4 corners.
    rect_set = pm.Data("rect_set", np.zeros(1, dtype="int64"))
    rect_group = pm.Data("rect_group", np.zeros(1, dtype="int64"))  # set (z) or n_sets + cell (numerator)
    rect_k = pm.Data("rect_k", np.ones(1))  # examples inside the block's ranges
    rect_log_x = pm.Data("rect_log_x", np.zeros(1))  # sum of log x over those examples
    rect_corner = pm.Data("rect_corner", np.zeros((1, 4), dtype="int64"))  # T indices: ++, -, -, +
    # Ranges holding no example share one likelihood: lumped per (set, family x span).
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 300)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_empty_size_count = pm.Data("cell_empty_size_count", np.zeros((1, 300)))  # empty ranges holding target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    rho = pm.Beta("rho", 1, 3)  # share of range prior on parity-restricted ranges
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)  # typical range span

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rules = pt.log(lambda_0) - np.log(N_RULES) + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)
    w_rules = pt.exp(log_rules - shift[:, None])

    # Compressed sampling weight within each family's range.
    x_w = pt.exp(-gamma * LOG_NUMBERS)[None, :] * FAMILY_MASK  # (3, 100)
    cum_w = pt.concatenate([pt.zeros((3, 1)), pt.cumsum(x_w, axis=1)], axis=1)
    range_w = cum_w[:, None, 1:] - cum_w[:, :-1, None]  # (3, a, b)
    log_w = pt.log(pt.where(VALID > 0, range_w, 1.0))
    log_fam_share = pt.log1p(-lambda_0) + pt.stack([pt.log1p(-rho), pt.log(rho / 2), pt.log(rho / 2)])
    # Erlang span prior, normalised within each family: log s - s / sigma - log Z_f.
    log_width = pt.log(SIZES) - SIZES / sigma  # (spans,)
    log_z = pm.math.logsumexp(LOG_FAM_SIZE_COUNT + log_width[None, :], axis=1, keepdims=True)  # (3, 1)
    log_size_prior = log_fam_share[:, None] + log_width[None, :] - log_z  # (3, spans)
    log_prior_grid = log_size_prior[:, SPAN_GRID]  # (3, a, b)
    # Range-specific part of each range's weight, for k = 1..4 examples inside.
    per_example = pt.exp(-beta * log_w)  # (3, a, b): (1 / W)^beta
    prior_grid = VALID * pt.exp(log_prior_grid)
    powers = [per_example]
    for _ in range(3):
        powers.append(powers[-1] * per_example)
    grid = pt.stack([prior_grid * pw for pw in powers], axis=1)  # (3, 4, a, b)
    table = pt.cumsum(pt.cumsum(grid[:, :, :, ::-1], axis=3)[:, :, :, ::-1], axis=2)
    table = pt.concatenate([pt.zeros((3, 4, 1, 100)), table], axis=2)
    table = pt.concatenate([table, pt.zeros((3, 4, 101, 1))], axis=3).reshape((-1,))
    corners = table[rect_corner]
    rect_sum = pt.maximum(corners[:, 0] - corners[:, 1] - corners[:, 2] + corners[:, 3], 0.0)
    block = beta * (
        rect_k * pt.log1p(-eps) - gamma * rect_log_x + (set_n[rect_set] - rect_k) * log_noise
    ) - shift[rect_set]
    w_rects = _scatter_sum(n_sets + n_cells, rect_group, pt.exp(block) * rect_sum)

    size_prior = pt.exp(log_size_prior).reshape((300,))
    empty_scale = pt.exp(beta * set_n * log_noise - shift)  # (sets,): shared likelihood of empty ranges
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)

    z = w_rules.sum(axis=1) + w_rects[:n_sets] + w_empty
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
        + w_rects[n_sets:]
        + empty_scale[cell_set] * (cell_empty_size_count * size_prior[None, :]).sum(axis=1)
    )
    in_concept = numer / z[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _blocks(examples, f):
    """Rectangles (a1, a2, b1, b2, k, sum log x) of family-f ranges holding >= 1 example."""
    ex = sorted(x for x in examples if FAMILY_MASK[f, x - 1] > 0)
    vals = sorted(set(ex))
    out = []
    for j in range(len(vals)):
        for l in range(j, len(vals)):
            inside = [x for x in ex if vals[j] <= x <= vals[l]]
            a1 = vals[j - 1] + 1 if j > 0 else 1
            b2 = vals[l + 1] - 1 if l + 1 < len(vals) else 100
            out.append((a1, vals[j], vals[l], b2, len(inside), float(np.log(inside).sum())))
    return out


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}
    n_sets = len(unique)

    rect_set, rect_group, rect_k, rect_log_x, rect_corner = [], [], [], [], []

    def add(i, group, f, a1, a2, b1, b2, k, slx):
        rect_set.append(i)
        rect_group.append(group)
        rect_k.append(float(k))
        rect_log_x.append(slx)
        rect_corner.append([
            table_index(f, k, a2, b1),
            table_index(f, k, a1 - 1, b1),
            table_index(f, k, a2, b2 + 1),
            table_index(f, k, a1 - 1, b2 + 1),
        ])

    set_blocks, empty_size_count, set_empty = [], [], []
    for i, st in enumerate(unique):
        ex = np.array(st)
        blocks = [(f, b) for f in range(3) for b in _blocks(st, f)]
        for f, (a1, a2, b1, b2, k, slx) in blocks:
            add(i, i, f, a1, a2, b1, b2, k, slx)
        set_blocks.append(blocks)
        inside = (
            (CON_A[:, None] <= ex[None, :])
            & (CON_B[:, None] >= ex[None, :])
            & (FAMILY_MASK[CON_F][:, ex - 1] > 0)
        )
        empty = np.nonzero(inside.sum(axis=1) == 0)[0]
        empty_size_count.append(np.bincount(CON_F[empty] * 100 + CON_S[empty], minlength=300).astype(float))
        set_empty.append(empty)

    cell_empty_size_count = []
    for c, (i, t) in enumerate(cells):
        for f, (a1, a2, b1, b2, k, slx) in set_blocks[i]:
            if a1 <= t <= b2 and FAMILY_MASK[f, t - 1] > 0:  # block ranges that also hold the target
                add(i, n_sets + c, f, a1, min(a2, t), max(b1, t), b2, k, slx)
        empty = set_empty[i]
        e_has = (CON_A[empty] <= t) & (CON_B[empty] >= t) & (FAMILY_MASK[CON_F[empty], t - 1] > 0)
        holds = empty[e_has]
        cell_empty_size_count.append(np.bincount(CON_F[holds] * 100 + CON_S[holds], minlength=300).astype(float))

    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "rect_set": np.array(rect_set, dtype="int64"),
        "rect_group": np.array(rect_group, dtype="int64"),
        "rect_k": np.array(rect_k, dtype="float64"),
        "rect_log_x": np.array(rect_log_x, dtype="float64"),
        "rect_corner": np.array(rect_corner, dtype="int64"),
        "empty_size_count": np.stack(empty_size_count),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_empty_size_count": np.stack(cell_empty_size_count),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
