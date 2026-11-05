"""Pedagogical sampling concept learning.

Hypothesis: people assume the examples were chosen by a helpful teacher, not
sampled at random: the program picked numbers that would best lead a learner to
its concept. So an example counts as evidence for a concept (one of 45
mathematical/digit rules, each with its own prior weight, or a range of 1-100
whose prior grows as a power of its size) in proportion to how strongly that
example alone would make a naive learner (one who assumes random sampling and
finds all rules and all ranges equally plausible) think of that concept,
compared with the other numbers 1-100 the teacher could have shown:
P_teach(x | h) is proportional to P_naive(h | x) ** teach. People answer "yes"
with their resulting belief that the concept contains the target.

Implementation note: for a range h, P_naive(h | x) takes one value for x inside
h and another outside, divided by the naive evidence Z(x) of x, so the teacher's
normaliser over x is computed exactly from a cumulative sum over 1-100, and a
set's evidence for a range depends only on how many examples fall inside it,
which is constant on rectangles of the (lower end, upper end) grid of ranges,
so sums over ranges are read off 2-D prefix sums.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
ALPHA_NOISE = 0.9
LOG_OUT = np.log((1 - ALPHA_NOISE) / 100)


def _rules():
    n = NUMBERS
    r = [n % 2 == 1, n % 2 == 0,
         np.array([k > 1 and all(k % a for a in range(2, int(k ** 0.5) + 1)) for k in n]),
         np.isin(n, [k * k for k in range(1, 11)]), np.isin(n, [1, 8, 27, 64]),
         np.isin(n, [2, 4, 8, 16, 64]), np.isin(n, [2, 4, 8, 16, 32, 37, 64])]
    r += [n % m == 0 for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    r += [np.isin(n, [m ** k for k in range(1, 8)]) for m in range(2, 11)]
    r += [n % 10 == d for d in range(10)]
    r += [np.array([str(d) in str(k) for k in n]) for d in range(10)]
    return np.array(r, dtype="float64")


RULES = _rules()
N_RULES = RULES.shape[0]
INT_LO = np.array([a for a in range(1, 101) for b in range(a, 101)])
INT_HI = np.array([b for a in range(1, 101) for b in range(a, 101)])
N_INT = INT_LO.size
INTERVALS = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")
INT_SIZE = (INT_HI - INT_LO + 1).astype("float64")
LOG_INT_SIZE = np.log(INT_SIZE)
INT_IN_LL = np.log(ALPHA_NOISE / INT_SIZE + (1 - ALPHA_NOISE) / 100)
RULE_LL = np.log(ALPHA_NOISE * RULES / RULES.sum(1, keepdims=True) + (1 - ALPHA_NOISE) / 100)
# Naive learner (flat prior: half rules, half ranges): log evidence Z(x) of one example.
_a = np.concatenate([np.log(0.5 / N_RULES) + RULE_LL,
                     np.log(0.5 / N_INT) + np.where(INTERVALS > 0, INT_IN_LL[:, None], LOG_OUT)])
_m = _a.max(axis=0)
LOG_Z = _m + np.log(np.exp(_a - _m).sum(axis=0))
LOG_Z = LOG_Z - LOG_Z.min()  # a constant shift cancels in the teacher's softmax
# A set of n examples supports a range only through k = how many examples it
# contains, so range evidence lives in 14 (n, k) slots of N_INT values each.
SLOT_OFF = {1: 0, 2: 2, 3: 5, 4: 9}
SLOT_N = np.array([n for n in range(1, 5) for k in range(n + 1)], dtype="float64")
SLOT_K = np.array([k for n in range(1, 5) for k in range(n + 1)], dtype="float64")
N_SLOT = SLOT_N.size
GRID_POS = INT_LO * 101 + INT_HI  # range's cell on the padded (lo, hi) grid
FULL_RANGE = (INT_LO == 1) & (INT_HI == 100)
LOWER_TRI = np.tril(np.ones((101, 101)))


def _cnt_lt(xs, lo):
    return sum(x < lo for x in xs)


def _cnt_le(xs, hi):
    return sum(x <= hi for x in xs)


def _rectangles(xs, t=None):
    """Rectangles of (lo, hi) on which a range contains exactly k of the examples
    (and, for a cell, the target t): yields (slot, lo_a, lo_b, hi_a, hi_b)."""
    lo_vals = np.arange(1, 101) if t is None else np.arange(1, t + 1)
    hi_vals = np.arange(1, 101) if t is None else np.arange(t, 101)
    i_of = np.array([_cnt_lt(xs, v) for v in lo_vals])
    j_of = np.array([_cnt_le(xs, v) for v in hi_vals])
    for i in np.unique(i_of):
        lo_sel = lo_vals[i_of == i]
        for j in np.unique(j_of):
            if j < i:
                continue
            hi_sel = hi_vals[j_of == j]
            yield SLOT_OFF[len(xs)] + (j - i), lo_sel[0], lo_sel[-1], hi_sel[0], hi_sel[-1]


def _corners(rects):
    """Signed indices into the flattened 2-D prefix sums P[lo, slot, hi] (0 = empty)."""
    out = []
    for slot, la, lb, ha, hb in rects:
        row = N_SLOT * 101  # prefix sums are laid out as (lo, slot, hi)

        def at(lo, hi):
            return lo * row + slot * 101 + hi
        out += [(at(lb, hb), 1.0), (at(la - 1, hb), -1.0),
                (at(lb, ha - 1), -1.0), (at(la - 1, ha - 1), 1.0)]
    return out


def prepare_observed(rows):
    sets = [str(r["number_set"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    ex_idx = np.full((n_sets, 4), 100, dtype="int64")  # 100 = padding
    set_xs = []
    for s, i in index.items():
        xs = [int(x) for x in s.split(",")]
        ex_idx[i, :len(xs)] = [x - 1 for x in xs]
        set_xs.append(xs)
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_of = {c: j for j, c in enumerate(cells)}
    # One row per set (all ranges: the normaliser), then one per cell (ranges containing the target).
    row_corners = [_corners(_rectangles(xs)) for xs in set_xs]
    row_corners += [_corners(_rectangles(set_xs[i], t + 1)) for (i, t) in cells]
    width = max(len(c) for c in row_corners)
    c_idx = np.zeros((len(row_corners), width), dtype="int64")
    c_sign = np.zeros((len(row_corners), width), dtype="float64")
    for r, cs in enumerate(row_corners):
        c_idx[r, :len(cs)] = [c for c, _ in cs]
        c_sign[r, :len(cs)] = [g for _, g in cs]
    return {
        "example_idx": ex_idx,
        "cell_set": np.array([c[0] for c in cells], dtype="int64"),
        "cell_target": np.array([c[1] for c in cells], dtype="int64"),
        "corner_idx": c_idx,
        "corner_sign": c_sign,
        "row_cell": np.array([cell_of[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    example_idx = pm.Data("example_idx", np.full((1, 4), 100, dtype="int64"))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    corner_idx = pm.Data("corner_idx", np.zeros((2, 4), dtype="int64"))
    corner_sign = pm.Data("corner_sign", np.zeros((2, 4), dtype="float64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    n_sets = example_idx.shape[0]

    # Learner's prior: share of rules, each rule's weight, range-size power.
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    size_power = pm.Normal("size_power", 0.0, 1.0)
    # How sharply the teacher prefers the most informative example.
    teach = pm.LogNormal("teach", 0.0, 0.75)

    log_int = size_power * LOG_INT_SIZE
    log_int = log_int - pt.logsumexp(log_int) + pt.log(1 - lambda_0)
    log_rule = pt.log(lambda_0) + pt.log(rule_weights)

    # Teacher, rules: log P_teach(x | rule) over x = 1..100, shape (100, R).
    zr = teach * (RULE_LL - LOG_Z[None, :]).T
    rule_ex_ll = zr - pt.logsumexp(zr, axis=0, keepdims=True)
    rule_ex_pad = pt.concatenate([rule_ex_ll, pt.zeros((1, N_RULES))], axis=0)
    # Set-constant term -teach * sum log Z(examples) is dropped from the ranges, so add it back here.
    ex_logz = pt.concatenate([pt.as_tensor(LOG_Z), pt.zeros(1)])[example_idx].sum(axis=1)
    rule_joint = log_rule[None, :] + rule_ex_pad[example_idx].sum(axis=1) + teach * ex_logz[:, None]

    # Teacher, ranges: normaliser from cumulative sums of exp(-teach * log Z(x)).
    w = pt.exp(-teach * LOG_Z)
    cum = pt.concatenate([pt.zeros(1), pt.cumsum(w)])
    w_in = cum[INT_HI] - cum[INT_LO - 1]
    # The range 1-100 has no outside numbers: give it a dummy outside mass, then drop that term.
    w_out = pt.where(FULL_RANGE, 1.0, cum[INT_LO - 1] + (cum[100] - cum[INT_HI]))
    log_in = teach * INT_IN_LL + pt.log(w_in)
    log_norm = pt.where(FULL_RANGE, log_in,
                        pt.logaddexp(log_in, teach * LOG_OUT + pt.log(w_out)))
    # Log prior x teacher likelihood of n examples, k of them inside the range: (slot, range).
    log_v = (log_int[None, :] - SLOT_N[:, None] * log_norm[None, :]
             + teach * (SLOT_K[:, None] * INT_IN_LL[None, :] + (SLOT_N - SLOT_K)[:, None] * LOG_OUT))
    shift = pt.max(log_v)
    v = pt.exp(log_v - shift)  # (slot, range)
    # Lay each slot on the (lo, hi) grid and take 2-D prefix sums: P[slot, lo, hi].
    # Grid laid out as (lo, slot, hi); hi-prefix by cumsum, lo-prefix by a triangular product.
    grid = pt.zeros((N_SLOT, 101 * 101))[:, GRID_POS].set(v).reshape((N_SLOT, 101, 101))
    grid = pt.cumsum(grid, axis=2).transpose(1, 0, 2).reshape((101, N_SLOT * 101))
    prefix = pt.dot(LOWER_TRI, grid).flatten()
    sums = (prefix[corner_idx] * corner_sign).sum(axis=1)
    log_int_den = pt.log(sums[:n_sets] + 1e-300) + shift  # (S,)
    int_num = sums[n_sets:]  # (C,), scaled by exp(-shift)

    log_den = pt.logaddexp(pt.logsumexp(rule_joint, axis=1), log_int_den)  # (S,)
    rule_post = pt.exp(rule_joint - log_den[:, None])  # (S, R)
    p_rule = (rule_post[cell_set] * pt.as_tensor(RULES.T)[cell_target]).sum(axis=1)
    p_int = int_num * pt.exp(shift - log_den[cell_set])
    p_yes = pm.Deterministic("p_yes", pt.clip((p_rule + p_int)[row_cell], 1e-6, 1 - 1e-6))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
