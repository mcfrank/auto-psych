"""Volatile-concept updating over log-compressed magnitude ranges.

Refinement of log_compressed_range_sampling: people do the same Bayesian concept
learning (42 weighted rules with a rule-size prior and all ranges of 1..100,
10% noisy examples, evidence tempered separately for rules and ranges, the first
example's evidence weighted by an inferred multiplier, a decision gain and
shifted criterion), and they think of ranges on a compressed mental number line:
an example x inside range [lo, hi] is sampled with probability
x ** -compression / sum_{k=lo..hi} k ** -compression (an example outside a range
is the 10% uniform noise). The one change, taken from volatile_concept_changepoint,
is that the examples are read one at a time and the program is believed to
switch concept between outputs: before each new example the concept is re-drawn
from the prior with an inferred switch probability, so recent examples count
more. Zero switch probability recovers log_compressed_range_sampling.

Computation: after the changepoint recursion, a range's belief is a sum over
"last switch before example j" terms, each the prior times its evidence since j.
Within one set, ranges containing the same examples form a rectangle of
(lo, hi) cells that share every term except a power of the compressed width,
so each term's mass over a rectangle (and over its ranges covering a target) is
read from summed-area tables of width ** -k; the filter is exact.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
MAX_SET = 4
MAX_CELLS = (MAX_SET + 1) * (MAX_SET + 2) // 2
N_CLASSES = 1 + 2 * MAX_SET  # (examples inside since the switch, first example among them)
CLASS_N = np.array([0] + [n for n in range(1, MAX_SET + 1) for a in (0, 1)], dtype="float64")
CLASS_A = np.array([0] + [a for n in range(1, MAX_SET + 1) for a in (0, 1)], dtype="float64")
SIGNS = np.array([1.0, -1.0, -1.0, 1.0])
EMPTY = -1e4  # log-weight of a term that does not exist


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
LO_GRID, HI_GRID = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
VALID = LO_GRID <= HI_GRID
LOG_N_RANGES = np.log(VALID.sum())
LOG_IN, LOG_OUT = np.log(1 - EPSILON), np.log(EPSILON / 100)

with pm.Model() as model:
    rule_step_lik = pm.Data("rule_step_lik", np.zeros((1, MAX_SET, N_RULES)))  # (sets, step, rules)
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_SET)))  # 1 where the step is a real example
    step_logx = pm.Data("step_logx", np.zeros((1, MAX_SET)))  # log of each example
    cell_in = pm.Data("cell_in", np.zeros((1, MAX_SET, MAX_CELLS)))  # example k inside the cell's ranges
    cell_mask = pm.Data("cell_mask", np.zeros((1, MAX_CELLS)))
    # Per (set, step k, last switch j, cell): class and summed-area corners of its term.
    term_class = pm.Data("term_class", np.zeros((1, MAX_SET, MAX_SET, MAX_CELLS), dtype="int64"))
    term_corners = pm.Data("term_corners", np.zeros((1, MAX_SET, MAX_SET, MAX_CELLS, 4), dtype="int64"))
    # Per distinct (set, target): corners of each final term's ranges covering the target.
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_target = pm.Data("pair_target", np.zeros(1, dtype="int64"))
    pair_class = pm.Data("pair_class", np.zeros((1, MAX_SET, MAX_CELLS), dtype="int64"))
    pair_corners = pm.Data("pair_corners", np.zeros((1, MAX_SET, MAX_CELLS, 4), dtype="int64"))
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    compression = pm.Normal("compression", mu=0.0, sigma=1.0)
    # The single change: believed probability that the program switches concept
    # (re-drawn from the prior) between consecutive outputs.
    switch_prob = pm.Beta("switch_prob", 1.5, 6.0)
    log_stay = pt.log1p(-switch_prob)
    log_switch = pt.log(switch_prob)

    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = pt.log(p_mix) + log_rule_prior - pt.logsumexp(log_rule_prior)
    log_range_prior = pt.log1p(-p_mix) - LOG_N_RANGES

    # Compressed width of every range, and tables of width ** -k per class.
    cum = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-compression * LOG_NUMBERS))])
    log_z = pt.log(pt.switch(VALID, cum[HI_GRID + 1] - cum[LO_GRID], 1.0))
    log_z_min = pt.min(pt.switch(VALID, log_z, np.inf))
    log_z = log_z - log_z_min
    k_class = range_evidence_weight * (CLASS_N + (anchor_weight - 1.0) * CLASS_A)  # (classes,)
    q = pt.exp(-k_class[:, None, None] * log_z[None]) * VALID[None]
    t = pt.cumsum(q, axis=1)
    t = pt.cumsum(t[:, :, ::-1], axis=2)[:, :, ::-1]
    t = pt.concatenate([pt.zeros((N_CLASSES, 1, 100)), t], axis=1)
    t = pt.concatenate([t, pt.zeros((N_CLASSES, 101, 1))], axis=2).reshape((-1,))

    def log_rect(corners, cls):
        s = pt.maximum((t[corners] * SIGNS).sum(axis=-1), 1e-300)
        return pt.log(s) - k_class[cls] * log_z_min

    cell_off = (cell_mask - 1.0) * 1e4
    rules_lp, terms = None, []
    for k in range(MAX_SET):
        w = anchor_weight if k == 0 else 1.0
        rule_ev = w * rule_evidence_weight * rule_step_lik[:, k, :]
        inside = cell_in[:, k, :]
        # Evidence of example k for a cell's ranges, apart from the width term.
        a_k = w * range_evidence_weight * (
            inside * (LOG_IN - compression * step_logx[:, k][:, None]) + (1 - inside) * LOG_OUT
        )
        if k == 0:
            rules_un = log_rule_prior[None, :] + rule_ev
            new_terms = [log_range_prior + a_k + cell_off]
        else:
            rules_un = pt.logaddexp(log_stay + rules_lp, log_switch + log_rule_prior[None, :]) + rule_ev
            new_terms = [L + log_stay + a_k for L in terms] + [log_switch + log_range_prior + a_k + cell_off]
        masses = [L + log_rect(term_corners[:, k, j], term_class[:, k, j]) for j, L in enumerate(new_terms)]
        log_norm = pt.logsumexp(pt.concatenate([rules_un] + masses, axis=1), axis=1, keepdims=True)
        rules_new = rules_un - log_norm
        new_terms = [L - log_norm for L in new_terms]
        if k == 0:
            rules_lp, terms = rules_new, new_terms
        else:
            m = step_mask[:, k][:, None]
            rules_lp = m * rules_new + (1 - m) * rules_lp
            old = terms + [pt.full_like(new_terms[-1], EMPTY)]
            terms = [m * L + (1 - m) * Lo for L, Lo in zip(new_terms, old)]

    rule_cover = (pt.exp(rules_lp)[pair_set] * pt.as_tensor(RULE_MEMBERSHIP.T)[pair_target]).sum(axis=1)
    range_cover = sum(
        pt.exp(L[pair_set] + log_rect(pair_corners[:, j], pair_class[:, j])).sum(axis=1)
        for j, L in enumerate(terms)
    )
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
    for l1, l2 in lo_seg:
        for h1, h2 in hi_seg:
            if l1 > l2 or h1 > h2 or l1 > h2:
                continue  # no valid range
            inside = [p for p in range(len(s)) if l2 <= s[p] - 1 <= h1]
            cells.append((l1, l2, h1, h2, inside))
    return cells


def _class(inside, j, k):
    """Class of the term 'last switch before example j' after example k."""
    n = sum(1 for p in inside if j <= p <= k)
    a = 1 if (j == 0 and 0 in inside) else 0
    return 0 if n == 0 else 1 + (n - 1) * 2 + a


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    tgts = [int(r["target"]) - 1 for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    pairs = list(dict.fromkeys(zip(sets, tgts)))
    pair_index = {p: i for i, p in enumerate(pairs)}
    n_sets, n_pairs = len(unique), len(pairs)
    empty = _corners(0, 0, -1, 0, -1)
    out = {
        "rule_step_lik": np.zeros((n_sets, MAX_SET, N_RULES)),
        "step_mask": np.zeros((n_sets, MAX_SET)),
        "step_logx": np.zeros((n_sets, MAX_SET)),
        "cell_in": np.zeros((n_sets, MAX_SET, MAX_CELLS)),
        "cell_mask": np.zeros((n_sets, MAX_CELLS)),
        "term_class": np.zeros((n_sets, MAX_SET, MAX_SET, MAX_CELLS), dtype="int64"),
        "term_corners": np.tile(np.array(empty, dtype="int64"), (n_sets, MAX_SET, MAX_SET, MAX_CELLS, 1)),
        "pair_class": np.zeros((n_pairs, MAX_SET, MAX_CELLS), dtype="int64"),
        "pair_corners": np.tile(np.array(empty, dtype="int64"), (n_pairs, MAX_SET, MAX_CELLS, 1)),
    }
    cells_of = []
    for i, s in enumerate(unique):
        cells = _cells(s)
        cells_of.append(cells)
        for k, x in enumerate(s):
            out["rule_step_lik"][i, k] = RULE_EXAMPLE_LOG_LIK[:, x - 1]
            out["step_mask"][i, k] = 1.0
            out["step_logx"][i, k] = np.log(x)
        for c, (l1, l2, h1, h2, inside) in enumerate(cells):
            out["cell_mask"][i, c] = 1.0
            for p in inside:
                out["cell_in"][i, p, c] = 1.0
            for k in range(len(s)):
                for j in range(k + 1):
                    cls = _class(inside, j, k)
                    out["term_class"][i, k, j, c] = cls
                    out["term_corners"][i, k, j, c] = _corners(cls, l1, l2, h1, h2)
    for u, (s, tg) in enumerate(pairs):
        i, last = index[s], len(s) - 1
        for c, (l1, l2, h1, h2, inside) in enumerate(cells_of[i]):
            for j in range(len(s)):
                cls = _class(inside, j, last)
                out["pair_class"][u, j, c] = cls
                out["pair_corners"][u, j, c] = _corners(cls, l1, min(l2, tg), max(h1, tg), h2)
    out.update({
        "pair_set": np.array([index[s] for s, _ in pairs], dtype="int64"),
        "pair_target": np.array([tg for _, tg in pairs], dtype="int64"),
        "row_pair": np.array([pair_index[p] for p in zip(sets, tgts)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    })
    return out
