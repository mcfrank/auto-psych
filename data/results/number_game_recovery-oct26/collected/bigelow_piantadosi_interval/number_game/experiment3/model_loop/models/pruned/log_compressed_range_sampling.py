"""Log-compressed sampling of magnitude ranges.

People do the same Bayesian concept learning as rule_size_prior_anchored_evidence
(42 weighted rules with a rule-size prior and all ranges of 1..100, evidence
tempered separately for rules and ranges, the first example's evidence weighted
by an inferred multiplier, 10% noisy examples, a decision gain and shifted
criterion), but they think about magnitude ranges on a compressed mental number
line: they believe a range's examples are spread evenly in compressed space, so
a number x inside a range [lo, hi] is sampled with probability
x ** -compression / sum_{k=lo..hi} k ** -compression (compression 1 = a
logarithmic line, 0 = a linear one). A range's effective size is therefore its
compressed width: ranges among high numbers count as small, ranges among low
numbers as wide. (An example inside a range is explained by the range; one
outside it is the 10% uniform noise.)

Computation: within one set, ranges that contain the same examples form a
rectangle of (lo, hi) cells sharing every term except the compressed width, so
each rectangle's mass (and its mass over ranges covering a target) is read from
summed-area tables of width ** -k; the posterior is exact.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
MAX_SET = 4
MAX_CELLS = (MAX_SET + 1) * (MAX_SET + 2) // 2
N_CLASSES = 1 + 2 * MAX_SET  # (examples inside, first example inside)
CLASS_N = np.array([0] + [n for n in range(1, MAX_SET + 1) for a in (0, 1)], dtype="float64")
CLASS_A = np.array([0] + [a for n in range(1, MAX_SET + 1) for a in (0, 1)], dtype="float64")
SIGNS = np.array([1.0, -1.0, -1.0, 1.0])


def rule_sets():
    """42 rules."""
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    return rules


RULES = rule_sets()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EXAMPLE_LOG_LIK = np.log(
    RULE_MEMBERSHIP * (1 - EPSILON) / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100
)
RULE_LOG_SIZE_C = np.log(RULE_MEMBERSHIP.sum(axis=1))
RULE_LOG_SIZE_C = RULE_LOG_SIZE_C - RULE_LOG_SIZE_C.mean()
# Ranges on a 100 x 100 grid indexed (lo - 1, hi - 1); only lo <= hi is a range.
LO_GRID, HI_GRID = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
VALID = LO_GRID <= HI_GRID
LOG_N_RANGES = np.log(VALID.sum())
LOG_IN, LOG_OUT = np.log(1 - EPSILON), np.log(EPSILON / 100)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    rule_anchor_lik = pm.Data("rule_anchor_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    # Per set and cell (ranges containing the same examples): its class, the
    # log-magnitudes of the inside examples (all, first), the outside counts
    # (all, first), whether the cell exists, and its summed-area-table corners.
    cell_class = pm.Data("cell_class", np.zeros((1, MAX_CELLS), dtype="int64"))
    cell_logx = pm.Data("cell_logx", np.zeros((1, MAX_CELLS)))
    cell_logx_a = pm.Data("cell_logx_a", np.zeros((1, MAX_CELLS)))
    cell_out = pm.Data("cell_out", np.zeros((1, MAX_CELLS)))
    cell_out_a = pm.Data("cell_out_a", np.zeros((1, MAX_CELLS)))
    cell_mask = pm.Data("cell_mask", np.zeros((1, MAX_CELLS)))
    cell_corners = pm.Data("cell_corners", np.zeros((1, MAX_CELLS, 4), dtype="int64"))
    # Per distinct (set, target): corners of each cell's ranges that cover the target.
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_target = pm.Data("pair_target", np.zeros(1, dtype="int64"))
    pair_corners = pm.Data("pair_corners", np.zeros((1, MAX_CELLS, 4), dtype="int64"))
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    extra = anchor_weight - 1.0
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    # The single change: compression of the mental number line for ranges.
    compression = pm.Normal("compression", mu=0.0, sigma=1.0)

    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    rule_ev = rule_log_lik + extra * rule_anchor_lik
    log_rules = pt.log(p_mix) + log_rule_prior[None, :] + rule_evidence_weight * rule_ev

    # Compressed width of every range: sum of k ** -compression over its members.
    cum = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-compression * LOG_NUMBERS))])
    log_z = pt.log(pt.switch(VALID, cum[HI_GRID + 1] - cum[LO_GRID], 1.0))
    log_z_min = pt.min(pt.switch(VALID, log_z, np.inf))
    log_z = log_z - log_z_min  # for range; added back per cell below
    # Each inside example contributes width ** -1 (tempered, anchored): per class
    # an exponent k; tables of width ** -k summed over lo <= i and hi >= j.
    k = range_evidence_weight * (CLASS_N + extra * CLASS_A)  # (classes,)
    q = pt.exp(-k[:, None, None] * log_z[None]) * VALID[None]
    t = pt.cumsum(q, axis=1)
    t = pt.cumsum(t[:, :, ::-1], axis=2)[:, :, ::-1]
    t = pt.concatenate([pt.zeros((N_CLASSES, 1, 100)), t], axis=1)
    t = pt.concatenate([t, pt.zeros((N_CLASSES, 101, 1))], axis=2).reshape((-1,))

    def rect_sum(corners):
        return pt.maximum((t[corners] * SIGNS).sum(axis=-1), 0.0)

    cell_full = rect_sum(cell_corners)  # (sets, cells)
    n_in, a_in = pt.as_tensor(CLASS_N)[cell_class], pt.as_tensor(CLASS_A)[cell_class]
    cell_ev = (
        -compression * (cell_logx + extra * cell_logx_a)
        + (n_in + extra * a_in) * LOG_IN
        + (cell_out + extra * cell_out_a) * LOG_OUT
    )
    log_cells = (
        pt.log1p(-p_mix) - LOG_N_RANGES + range_evidence_weight * cell_ev
        - k[cell_class] * log_z_min + pt.log(pt.maximum(cell_full, 1e-300))
        + (cell_mask - 1.0) * 1000.0
    )
    log_norm = pt.logsumexp(pt.concatenate([log_rules, log_cells], axis=1), axis=1, keepdims=True)
    rule_post = pt.exp(log_rules - log_norm)  # (sets, rules)
    cell_post = pt.exp(log_cells - log_norm) * cell_mask  # (sets, cells)
    rule_cover = (rule_post[pair_set] * pt.as_tensor(RULE_MEMBERSHIP.T)[pair_target]).sum(axis=1)
    share = rect_sum(pair_corners) / pt.maximum(cell_full[pair_set], 1e-300)
    range_cover = (cell_post[pair_set] * share).sum(axis=1)
    concept_p = pt.clip((rule_cover + range_cover)[row_pair], 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _corners(cls, l1, l2, h1, h2):
    """Flat table indices (with signs SIGNS) of the sum over lo in [l1, l2], hi in [h1, h2]."""
    if l2 < l1 or h2 < h1:
        l2, h2 = l1 - 1, h1 - 1  # empty: the four corners cancel exactly
    base = cls * 101 * 101
    return [base + (l2 + 1) * 101 + h1, base + l1 * 101 + h1,
            base + (l2 + 1) * 101 + h2 + 1, base + l1 * 101 + h2 + 1]


def _cells(s):
    """Rectangles of (lo, hi) (0-based) whose ranges contain the same examples."""
    v = sorted(set(x - 1 for x in s))
    m = len(v)
    lo_seg = [(0 if a == 0 else v[a - 1] + 1, v[a] if a < m else 99) for a in range(m + 1)]
    hi_seg = [(v[b] if b >= 0 else 0, v[b + 1] - 1 if b + 1 < m else 99) for b in range(-1, m)]
    cells = []
    for a, (l1, l2) in enumerate(lo_seg):
        for b, (h1, h2) in zip(range(-1, m), hi_seg):
            if l1 > l2 or h1 > h2 or l1 > h2:
                continue  # no valid range
            inside = [p for p in range(len(s)) if l2 <= s[p] - 1 <= h1]
            cells.append((l1, l2, h1, h2, inside))
    return cells


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    tgts = [int(r["target"]) - 1 for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    pairs = list(dict.fromkeys(zip(sets, tgts)))
    pair_index = {p: i for i, p in enumerate(pairs)}
    n_sets = len(unique)
    out = {
        "cell_class": np.zeros((n_sets, MAX_CELLS), dtype="int64"),
        "cell_logx": np.zeros((n_sets, MAX_CELLS)),
        "cell_logx_a": np.zeros((n_sets, MAX_CELLS)),
        "cell_out": np.zeros((n_sets, MAX_CELLS)),
        "cell_out_a": np.zeros((n_sets, MAX_CELLS)),
        "cell_mask": np.zeros((n_sets, MAX_CELLS)),
        "cell_corners": np.zeros((n_sets, MAX_CELLS, 4), dtype="int64"),
        "pair_corners": np.zeros((len(pairs), MAX_CELLS, 4), dtype="int64"),
    }
    cells_of = []
    for i, s in enumerate(unique):
        cells = _cells(s)
        cells_of.append(cells)
        for j, (l1, l2, h1, h2, inside) in enumerate(cells):
            a = 1 if 0 in inside else 0
            n = len(inside)
            cls = 0 if n == 0 else 1 + (n - 1) * 2 + a
            out["cell_class"][i, j] = cls
            out["cell_logx"][i, j] = sum(np.log(s[p]) for p in inside)
            out["cell_logx_a"][i, j] = a * np.log(s[0])
            out["cell_out"][i, j] = len(s) - n
            out["cell_out_a"][i, j] = 1 - a
            out["cell_mask"][i, j] = 1.0
            out["cell_corners"][i, j] = _corners(cls, l1, l2, h1, h2)
    for u, (s, tg) in enumerate(pairs):
        i = index[s]
        for j, (l1, l2, h1, h2, inside) in enumerate(cells_of[i]):
            cls = out["cell_class"][i, j]
            out["pair_corners"][u, j] = _corners(cls, l1, min(l2, tg), max(h1, tg), h2)
    out.update({
        "rule_log_lik": np.stack([RULE_EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "rule_anchor_lik": np.stack([RULE_EXAMPLE_LOG_LIK[:, s[0] - 1] for s in unique]),
        "pair_set": np.array([index[s] for s, _ in pairs], dtype="int64"),
        "pair_target": np.array([tg for _, tg in pairs], dtype="int64"),
        "row_pair": np.array([pair_index[p] for p in zip(sets, tgts)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    })
    return out
