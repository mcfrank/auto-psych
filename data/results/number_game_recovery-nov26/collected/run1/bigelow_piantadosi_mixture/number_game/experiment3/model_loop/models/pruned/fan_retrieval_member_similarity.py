"""Fan-effect retrieval of stored number categories, endorsement by member similarity.

People do not infer the program's concept; they retrieve familiar number
categories from memory -- 45 rules ("even", "powers of 2", "ends in 6", ...) and
stretches of the number line (every interval of 1-100, with a gamma-shaped prior
familiarity over widths, shape and rate fitted) -- each stored as the list of its
member numbers. Each example cues the categories that store it, and a category
is retrieved only if at least one example cues it; by the memory fan effect each
cue's strength is diluted by the number of members the category stores (1 / size
per cueing example), and an example the category does not store counts as a
stray number (fitted trust `alpha`, as in the size-principle learners). Digit rules carry a fitted familiarity weight.
The target is endorsed by its similarity to the nearest stored member of each
retrieved category, exp(-c * distance) on the number line (c fitted), weighted by
retrieval strength: members are endorsed fully, near-misses partly. All examples
cue memory together, so order plays no role.

Interval sums are done in closed form: the examples an interval holds form a
contiguous run of the sorted examples, and for each set, run, width and target
the placements holding exactly that run give consecutive edge distances, whose
summed similarity is a geometric series in exp(-c).
"""

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt
import scipy.sparse as sparse

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


RULE_NAMES = list(rule_concepts().keys())
RULES = list(rule_concepts().values())
IS_DIGIT_RULE = np.array([n.startswith("ends in") or n.startswith("contains digit") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# Distance from each number to the nearest member of each rule.
RULE_DMIN = np.array([[min(abs(t - m) for m in c) for t in NUMBERS] for c in RULES], dtype="float64")
WIDTHS = np.arange(1, 101, dtype="float64")
LOG_WIDTH_COUNT = np.log(101 - WIDTHS)

MAX_RUNS = 10  # contiguous runs of up to 4 sorted examples
SLOTS = 103  # per pair: placements containing the target, then edge distances 0..101

with pm.Model() as model:
    rule_hits = pm.Data("rule_hits", np.ones((1, N_RULES)))  # examples the rule stores
    n_examples = pm.Data("n_examples", np.ones(1))
    # Cells: (set, run of examples, width) with placements holding exactly that run.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_k = pm.Data("cell_k", np.ones(1))  # examples in the run
    cell_w = pm.Data("cell_w", np.zeros(1, dtype="int64"))  # width - 1
    cell_log_place = pm.Data("cell_log_place", np.zeros(1))  # log #such placements
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_target = pm.Data("pair_target", np.zeros(1, dtype="int64"))  # target - 1
    # Sparse map (CSR) from per-interval strengths of each (set, run, width) to, for
    # each pair, the count of placements containing the target (slot 0) and the
    # +1 / -1 endpoints of the geometric series of edge distances (slot 1 + d).
    csr_data = pm.Data("csr_data", np.zeros(1))
    csr_indices = pm.Data("csr_indices", np.zeros(1, dtype="int32"))
    csr_indptr = pm.Data("csr_indptr", np.zeros(SLOTS + 1, dtype="int32"))
    csr_shape = pm.Data("csr_shape", np.array([SLOTS, 1], dtype="int32"))
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)  # share of familiarity on rules
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    rule_lw = log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
    rule_log_prior = rule_lw - pt.logsumexp(rule_lw)

    # Gamma-shaped familiarity of interval widths (rate parameterisation).
    shape_m1 = pm.Normal("shape_m1", mu=0.5, sigma=1.0)
    rate_offset = pm.Normal("rate_offset", mu=0.0, sigma=1.5)
    rate = 2.4 * shape_m1 + rate_offset
    width_lp = shape_m1 * np.log(WIDTHS) - rate * WIDTHS / 100.0
    width_lp = width_lp - pt.logsumexp(LOG_WIDTH_COUNT + width_lp)  # per-interval log prior

    # Fan-effect retrieval strength: each cueing example adds alpha/size (+ stray
    # chance), each non-cueing example only the stray chance.
    alpha = pm.Beta("alpha", 9, 1)
    log_stray = pt.log1p(-alpha) - np.log(100.0)
    n = n_examples[:, None]
    rule_cue = pt.log(alpha / RULE_SIZE + (1 - alpha) / 100.0)[None, :]
    rule_ls = (pt.log(lambda_0) + rule_log_prior[None, :] + rule_hits * rule_cue + (n - rule_hits) * log_stray
               - 1e4 * pt.lt(rule_hits, 0.5))
    width_cue = pt.log(alpha / WIDTHS + (1 - alpha) / 100.0)  # (widths,)
    int_ls = (pt.log1p(-lambda_0) + width_lp[cell_w] + cell_k * width_cue[cell_w]
              + (n_examples[cell_set] - cell_k) * log_stray)  # per interval of each cell
    # Strengths stay far above float underflow (> exp(-100)), so normalise directly.
    rule_str = pt.exp(rule_ls)  # (sets, rules)
    int_str = pt.exp(int_ls)
    z = rule_str.sum(axis=1) + pt.zeros(rule_str.shape[0])[cell_set].inc(int_str * pt.exp(cell_log_place))
    rule_act = rule_str / z[:, None]
    int_act = int_str / z[cell_set]  # per interval

    # Similarity to the nearest stored member, exp(-c d).
    log_c = pm.Normal("log_c", mu=0.5, sigma=1.0)
    c = pt.exp(log_c)
    rule_sim = pt.exp(-c * pt.as_tensor(RULE_DMIN))  # (rules, 100)
    gen_rule = pt.dot(rule_act, rule_sim)[pair_set, pair_target]
    weights = pt.concatenate([pt.ones(1), pt.exp(-c * pt.arange(102)) / (-pt.expm1(-c))])  # (SLOTS,)
    to_pairs = ps.CSR(csr_data, csr_indices, csr_indptr, csr_shape)
    slots = ps.structured_dot(to_pairs, int_act[:, None]).reshape((-1, SLOTS))
    gen_int = pt.dot(slots, weights)

    p_yes = pm.Deterministic("p_yes", pt.clip((gen_rule + gen_int)[row_pair], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def runs_of(s):
    """Contiguous runs of the sorted distinct examples: (k, lo_min, hi_max, low bound, high bound)."""
    xs = sorted(set(s))
    out = []
    for i in range(len(xs)):
        for j in range(i, len(xs)):
            k = sum(1 for x in s if xs[i] <= x <= xs[j])
            prev = xs[i - 1] if i > 0 else 0
            nxt = xs[j + 1] if j + 1 < len(xs) else 101
            out.append((k, xs[i], xs[j], prev, nxt))
    return out


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    w = NUMBERS
    S = len(unique)
    rule_hits = np.zeros((S, N_RULES))
    run_k = np.zeros((S, MAX_RUNS))
    log_place = np.full((S, MAX_RUNS, 100), -1e4)
    lo_range = {}
    for i, s in enumerate(unique):
        rule_hits[i] = RULE_MEMBERSHIP[:, [x - 1 for x in s]].sum(axis=1)
        for r, (k, a, b, prev, nxt) in enumerate(runs_of(s)):
            # lower edge lo: holds a..b, excludes prev and nxt, stays in 1..100
            llo = np.maximum.reduce([np.ones(100, int), b - w + 1, np.full(100, prev + 1)])
            lhi = np.minimum.reduce([np.full(100, a), 101 - w, nxt - w])
            run_k[i, r] = k
            cnt = lhi - llo + 1
            log_place[i, r] = np.where(cnt > 0, np.log(np.maximum(cnt, 1)), -1e4)
            lo_range[i, r] = (llo, lhi, cnt > 0, a, b)
    pairs = list(dict.fromkeys(zip([index[s] for s in sets], targets)))
    pair_index = {p: j for j, p in enumerate(pairs)}
    P = len(pairs)
    n0 = np.zeros((P, MAX_RUNS, 100))
    d1 = np.full((P, MAX_RUNS, 100), 101, dtype="int64")
    d2p1 = np.full((P, MAX_RUNS, 100), 101, dtype="int64")
    for j, (i, t) in enumerate(pairs):
        for r in range(len(runs_of(unique[i]))):
            llo, lhi, ok, a, b = lo_range[i, r]
            if a <= t <= b:
                n0[j, r] = np.where(ok, lhi - llo + 1, 0)
                continue
            if t > b:
                hlo, hhi = llo + w - 1, lhi + w - 1
                n0[j, r] = np.where(ok, np.maximum(0, hhi - np.maximum(hlo, t) + 1), 0)
                top = np.minimum(hhi, t - 1)
                has = ok & (hlo <= top)
                d1[j, r] = np.where(has, t - top, 101)
                d2p1[j, r] = np.where(has, t - hlo + 1, 101)
            else:
                n0[j, r] = np.where(ok, np.maximum(0, np.minimum(lhi, t) - llo + 1), 0)
                bot = np.maximum(llo, t + 1)
                has = ok & (bot <= lhi)
                d1[j, r] = np.where(has, bot - t, 101)
                d2p1[j, r] = np.where(has, lhi - t + 1, 101)
    cs, cr, cw = np.nonzero(log_place > -1e3)
    cell_of = -np.ones((S, MAX_RUNS, 100), dtype="int64")
    cell_of[cs, cr, cw] = np.arange(len(cs))
    valid = (log_place[[p[0] for p in pairs]] > -1e3) & ((n0 > 0) | (d1 < 101))
    jj, rr, ww = np.nonzero(valid)
    cell = cell_of[np.array([p[0] for p in pairs])[jj], rr, ww]
    tail = d1[valid] < 101
    m_rows = np.concatenate([jj * SLOTS, (jj * SLOTS + 1 + d1[valid])[tail], (jj * SLOTS + 1 + d2p1[valid])[tail]])
    m_cols = np.concatenate([cell, cell[tail], cell[tail]])
    m_vals = np.concatenate([n0[valid], np.ones(tail.sum()), -np.ones(tail.sum())])
    to_pairs = sparse.csr_matrix((m_vals, (m_rows, m_cols)), shape=(P * SLOTS, len(cs)))
    to_pairs.sum_duplicates()
    to_pairs.sort_indices()
    return {
        "rule_hits": rule_hits,
        "n_examples": np.array([float(len(s)) for s in unique]),
        "cell_set": cs.astype("int64"),
        "cell_k": run_k[cs, cr],
        "cell_w": cw.astype("int64"),
        "cell_log_place": log_place[cs, cr, cw],
        "pair_set": np.array([p[0] for p in pairs], dtype="int64"),
        "pair_target": np.array([p[1] - 1 for p in pairs], dtype="int64"),
        "csr_data": to_pairs.data.astype("float64"),
        "csr_indices": to_pairs.indices.astype("int32"),
        "csr_indptr": to_pairs.indptr.astype("int32"),
        "csr_shape": np.array(to_pairs.shape, dtype="int32"),
        "row_pair": np.array([pair_index[(index[s], t)] for s, t in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
