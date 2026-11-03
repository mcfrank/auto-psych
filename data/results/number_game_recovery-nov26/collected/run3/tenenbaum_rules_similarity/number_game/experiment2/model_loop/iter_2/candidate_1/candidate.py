"""Pedagogical (teacher) sampling: people assume the examples were chosen by a
helpful teacher. A teacher who means concept h picks each example x in
proportion to how strongly x alone would lead a learner to h, i.e.
P_teach(x|h) ∝ P(h|x)^beta = [P(x|h) P(h) / P(x)]^beta, normalised over x in
1..100 (P(h) cancels). Examples that many other plausible concepts would also
produce (high learner marginal P(x)) are poor teaching examples, so all-multiples-
of-9 examples point to "multiples of 9" more strongly than strong sampling
implies. The concept space (45 rules with a family-organised prior and exception
rules as their own family, all intervals with a beta prior over the share of
1..100 they cover, examples genuine with probability 0.9) is the best model's;
the one change is the teacher-sampling likelihood with inferred teacher
optimality beta. "Yes" is the posterior probability the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
SAMPLER_SETTINGS = {"draws": 1000, "tune": 1000, "target_accept": 0.95}


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


def rule_family(name):
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers of 2,"):
        return 7
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_P = ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100
RULE_LOGP = np.log(RULE_P)  # strong-sampling log P(x|rule)
LO = np.array([m for m in range(100) for n in range(m, 100)])  # 0-based interval endpoints
HI = np.array([n for m in range(100) for n in range(m, 100)])
SIZE_IDX = HI - LO  # size - 1
SIZES = np.arange(1, 101, dtype="float64")
LOG_IN = np.log(ALPHA / SIZES + (1 - ALPHA) / 100)  # log P(x|interval of size L), x inside
LOG_OUT = np.log((1 - ALPHA) / 100)
LOG_N_OF_SIZE = np.log(101.0 - SIZES)
SHARE = (SIZES - 0.5) / 100.0
# Q[L, x] = sum over intervals of size L of P(x | interval): learner's marginal, per size
_INSIDE = (np.arange(100)[None, :] >= LO[:, None]) & (np.arange(100)[None, :] <= HI[:, None])
Q_SIZE = np.zeros((100, 100))
np.add.at(Q_SIZE, SIZE_IDX, np.where(_INSIDE, np.exp(LOG_IN)[SIZE_IDX][:, None], np.exp(LOG_OUT)))
N_EX_K = np.array([(n, k) for n in range(1, 5) for k in range(5)])  # (n examples, k inside) slots

FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)

# An interval's example count k is constant on rectangles of (low end, high end),
# so prepare_observed lists each set's (and each set+target's) rectangles as four
# signed corners into 2-D prefix sums of the per-interval weights.

with pm.Model() as model:
    rule_sll = pm.Data("rule_sll", np.zeros((1, N_RULES)))  # (sets, rules): sum_i log P(x_i|h)
    n_ex = pm.Data("n_ex", np.zeros(1))  # (sets,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    set_corner = pm.Data("set_corner", np.zeros(1, dtype="int64"))  # flat index into prefix sums
    set_corner_sign = pm.Data("set_corner_sign", np.zeros(1))
    set_corner_row = pm.Data("set_corner_row", np.zeros(1, dtype="int64"))
    cell_corner = pm.Data("cell_corner", np.zeros(1, dtype="int64"))
    cell_corner_sign = pm.Data("cell_corner_sign", np.zeros(1))
    cell_corner_row = pm.Data("cell_corner_row", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_share = pm.Beta("mean_share", 2, 3)
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # teacher optimality

    rule_log_prior = pt.log(lambda_0) + pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = pt.log1p(-lambda_0) + log_size_prior - LOG_N_OF_SIZE

    # Learner's marginal probability of a single example, P(x), under the prior.
    p_x = pt.exp(rule_log_prior) @ RULE_P + pt.exp(log_per_interval) @ Q_SIZE
    log_w = -beta * pt.log(p_x)  # teacher prefers examples rare under the marginal
    w = pt.exp(log_w - pt.max(log_w))  # common rescaling cancels in the posterior

    # Teacher normaliser Z_h = sum_x P(x|h)^beta P(x)^-beta.
    log_z_rule = pt.log(pt.exp(beta * RULE_LOGP) @ w)
    cum = pt.concatenate([pt.zeros(1), pt.cumsum(w)])
    s_in = cum[HI + 1] - cum[LO]
    s_out = pt.maximum(cum[-1] - s_in, 0.0)
    log_z_int = pt.log(pt.exp(beta * LOG_IN)[SIZE_IDX] * s_in + pt.exp(beta * LOG_OUT) * s_out)
    z_min = pt.minimum(pt.min(log_z_int), pt.min(log_z_rule))

    # Unnormalised posterior weight of each rule, per set.
    rule_w = pt.exp(rule_log_prior[None, :] + beta * rule_sll
                    - n_ex[:, None] * (log_z_rule - z_min)[None, :])
    # Weight of each interval for every (n, k) slot, laid out on a (low, high) grid.
    nn, kk = N_EX_K[:, 0][:, None], N_EX_K[:, 1][:, None]
    int_log_w = (log_per_interval[SIZE_IDX][None, :]
                 + beta * (kk * LOG_IN[SIZE_IDX][None, :] + (nn - kk) * LOG_OUT)
                 - nn * (log_z_int - z_min)[None, :])
    grid = pt.set_subtensor(pt.zeros((len(N_EX_K), 100, 100))[:, LO, HI], pt.exp(int_log_w))
    prefix = pt.cumsum(pt.cumsum(grid, axis=1), axis=2)
    prefix = pt.concatenate([pt.zeros((len(N_EX_K), 1, 100)), prefix], axis=1)
    prefix = pt.concatenate([pt.zeros((len(N_EX_K), 101, 1)), prefix], axis=2).flatten()

    set_int = pt.zeros(rule_sll.shape[0])[set_corner_row].inc(set_corner_sign * prefix[set_corner])
    cell_int = pt.zeros(cell_set.shape[0])[cell_corner_row].inc(cell_corner_sign * prefix[cell_corner])
    evidence = rule_w.sum(axis=1) + set_int
    cell_num = (rule_w[cell_set] * cell_rule_member).sum(axis=1) + cell_int
    in_concept = cell_num / evidence[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _segments(bounds):
    """Split 0..99 at the given cut points into [start, end] runs."""
    cuts = sorted(set(b for b in bounds if 0 < b < 100))
    edges = [0] + cuts + [100]
    return [(a, b - 1) for a, b in zip(edges[:-1], edges[1:])]


def _rect_corners(examples, t_lo, t_hi):
    """Corners (slot, row, col, sign) of rectangles of intervals with lo <= t_lo, hi >= t_hi."""
    ex = [e - 1 for e in examples]
    n = len(ex)
    out = []
    for a1, a2 in _segments([e + 1 for e in ex] + [t_lo + 1]):
        if a1 > t_lo:
            continue
        for b1, b2 in _segments(ex + [t_hi]):
            if b2 < t_hi or b2 < a1:
                continue
            k = sum(1 for e in ex if a1 <= e <= b1)
            slot = (n - 1) * 5 + k
            out += [(slot, a2 + 1, b2 + 1, 1.0), (slot, a1, b2 + 1, -1.0),
                    (slot, a2 + 1, b1, -1.0), (slot, a1, b1, 1.0)]
    return out


def _flatten(corner_lists):
    idx, sign, row = [], [], []
    for r, corners in enumerate(corner_lists):
        for slot, i, j, s in corners:
            idx.append(slot * 101 * 101 + i * 101 + j)
            sign.append(s)
            row.append(r)
    return np.array(idx, dtype="int64"), np.array(sign), np.array(row, dtype="int64")


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    s_idx, s_sign, s_row = _flatten([_rect_corners(s, 99, 0) for s in unique])
    c_idx, c_sign, c_row = _flatten([_rect_corners(unique[s], t - 1, t - 1) for s, t in cells])
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_sll": np.stack([RULE_LOGP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "n_ex": np.array([len(s) for s in unique], dtype="float64"),
        "cell_set": np.array([c[0] for c in cells], dtype="int64"),
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "set_corner": s_idx, "set_corner_sign": s_sign, "set_corner_row": s_row,
        "cell_corner": c_idx, "cell_corner_sign": c_sign, "cell_corner_row": c_row,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
