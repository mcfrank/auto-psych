"""Example-anchored exception neighbourhoods emitted through a separate side channel.

Refinement of `example_anchored_exception_neighbourhood`: the same 45 rules
(final-digit rules boosted by kappa), Erlang-width intervals with
compressed-number-line sampling, tempered size principle (beta), exception rate
eps, and "rule plus neighbourhood [x - h, x + h]" concepts anchored only on shown
examples x outside the rule (share alpha of the rule's prior, h = 0..3 with
P(h) proportional to exp(-rho h)). The one change, grafted from
`rule_with_exception_side_channel`: such a concept is believed to generate its
outputs as a main channel plus an occasional side channel -- at the program's
own slip rate eps it emits a member of the exception neighbourhood (uniformly
over its off-rule numbers), otherwise a rule member -- instead of sampling
uniformly from rule and neighbourhood pooled together. The side-channel rate is
not a separate parameter: exceptional outputs are pictured as the program's
occasional slips, landing around the outlier. A wider neighbourhood
then no longer dilutes the evidence the rule-following examples give, only the
outlier's own likelihood, so people credit a wider region around the outlier
and endorse its neighbours more than off-rule neighbours of majority examples.

Implementation note: intervals holding the same run of examples share their
likelihood up to a per-interval factor, so each such class is a rectangle of the
(a, b) grid whose summed weight is read off a 2D prefix-sum table.
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


RULE_NAMES = list(rule_concepts().keys())
RULES = list(rule_concepts().values())
# Rules read straight off the final digit: parity and "ends in d".
RULE_SALIENT = np.array([n in ("odd", "even") or n.startswith("ends in") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
# Dense interval grid: entry [a-1, b-1] is the interval [a, b], valid when b >= a.
GRID_A = np.repeat(NUMBERS, 100).reshape(100, 100)
GRID_B = np.tile(NUMBERS, 100).reshape(100, 100)
GRID_VALID = (GRID_B >= GRID_A).astype("float64")
GRID_WIDTH_IDX = np.clip(GRID_B - GRID_A, 0, 99)  # width - 1
H_MAX = 3  # largest exception-neighbourhood radius
H_RADII = np.arange(H_MAX + 1, dtype="float64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)  # intervals of each width
K_VALUES = np.arange(5, dtype="float64")  # examples an interval can hold (sets have at most four)


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    # Intervals holding >= 1 example fall into classes by which run of sorted examples they hold; each
    # class (or its part holding a target) is a rectangle a in [a0, a1], b in [b0, b1] of the (a, b) grid.
    rect_set = pm.Data("rect_set", np.zeros(1, dtype="int64"))
    rect_cell = pm.Data("rect_cell", np.zeros(1, dtype="int64"))  # owner cell, or -1 for the set total
    rect_k = pm.Data("rect_k", np.ones(1, dtype="int64"))  # examples inside
    rect_log_x = pm.Data("rect_log_x", np.zeros(1))  # sum of log x over those examples
    rect_a0 = pm.Data("rect_a0", np.ones(1, dtype="int64"))
    rect_a1 = pm.Data("rect_a1", np.ones(1, dtype="int64"))
    rect_b0 = pm.Data("rect_b0", np.ones(1, dtype="int64"))
    rect_b1 = pm.Data("rect_b1", np.ones(1, dtype="int64"))
    # Intervals holding no example all share one likelihood: counted by width.
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 100)))  # (sets, widths)
    cell_empty_size_count = pm.Data("cell_empty_size_count", np.zeros((1, 100)))  # empty intervals holding target
    # Cells (set, target).
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    # Rule-plus-neighbourhood concepts anchored on an off-rule example: one per (set, rule, centre, radius).
    cls_set = pm.Data("cls_set", np.zeros(1, dtype="int64"))
    cls_rule = pm.Data("cls_rule", np.zeros(1, dtype="int64"))
    cls_h = pm.Data("cls_h", np.zeros(1, dtype="int64"))
    cls_k = pm.Data("cls_k", np.zeros(1))
    cls_krule = pm.Data("cls_krule", np.zeros(1))  # examples inside the rule itself
    cls_esize = pm.Data("cls_esize", np.ones(1))  # off-rule numbers in the neighbourhood
    cls_log_m = pm.Data("cls_log_m", np.zeros(1))  # log of the off-rule examples the exception may be anchored on
    cls_n_out = pm.Data("cls_n_out", np.zeros(1))  # examples outside the whole concept
    # Sparse (cell, concept): concepts whose neighbourhood holds a target outside the rule.
    chit_cell = pm.Data("chit_cell", np.zeros(1, dtype="int64"))
    chit_cls = pm.Data("chit_cls", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)  # typical interval width
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)  # log prior boost of final-digit rules
    alpha = pm.Beta("alpha", 1, 4)  # share of a rule's prior given to its exception variants
    rho = pm.LogNormal("rho", mu=0.0, sigma=1.0)  # how fast prior falls with the exception's radius

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)  # (rules,)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rule_prior = kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    # A rule cedes the share alpha to anchored variants only if some example lies outside it.
    has_off = pt.gt(set_n[:, None] - rule_k, 0.5)
    log_rules = pt.log(lambda_0) + pt.log1p(-alpha * has_off) + log_rule_prior[None, :] + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)  # per-set stabiliser
    w_rules = pt.exp(log_rules - shift[:, None])

    # Rule R plus neighbourhood of radius h around a shown example x not in R, as two channels:
    # the neighbourhood's off-rule numbers at the program's slip rate eps, otherwise rule members.
    log_h_prior = -rho * H_RADII - pm.math.logsumexp(-rho * H_RADII)
    side_rule_in = pt.log((1 - eps) * (1 - eps) / RULE_SIZE + eps / 100)  # (rules,)
    side_exc_in = pt.log((1 - eps) * eps / cls_esize + eps / 100)
    log_cls = (
        pt.log(lambda_0) + pt.log(alpha) + log_rule_prior[cls_rule] + log_h_prior[cls_h]
        - cls_log_m
        + beta * (
            cls_krule * side_rule_in[cls_rule]
            + (cls_k - cls_krule) * side_exc_in
            + cls_n_out * log_noise
        )
    )
    w_cls = pt.exp(log_cls - shift[cls_set])
    w_aug = _scatter_sum(n_sets * N_RULES, cls_set * N_RULES + cls_rule, w_cls).reshape((n_sets, N_RULES))

    # Interval members are drawn with weight x^-gamma: P(x | [a, b]) = x^-gamma / W(a, b).
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    diff = cum_w[GRID_B] - cum_w[GRID_A - 1]
    log_w = pt.log(pt.where(GRID_VALID > 0, diff, 1.0))  # (100, 100)
    # Erlang width prior: log P(interval of width s) = log s - s / sigma - log Z.
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z  # (widths,)
    # g[k, a, b] = prior([a, b]) * W(a, b)^(-beta k): the part of an interval's weight that varies within a class.
    log_g = log_size_prior[GRID_WIDTH_IDX][None, :, :] - beta * K_VALUES[:, None, None] * log_w[None, :, :]
    g = pt.exp(log_g) * GRID_VALID[None, :, :]
    # table[k, a, b] = sum of g over a' <= a, b' >= b (padded: a = 0 and b = 101 give zero).
    tail_b = pt.cumsum(g[:, :, ::-1], axis=2)[:, :, ::-1]
    table = pt.cumsum(tail_b, axis=1)
    table = pt.concatenate([pt.zeros((K_VALUES.shape[0], 1, 100)), table], axis=1)
    table = pt.concatenate([pt.zeros((K_VALUES.shape[0], 101, 1)), table, pt.zeros((K_VALUES.shape[0], 101, 1))], axis=2)
    rect_g = (
        table[rect_k, rect_a1, rect_b0] - table[rect_k, rect_a0 - 1, rect_b0]
        - table[rect_k, rect_a1, rect_b1 + 1] + table[rect_k, rect_a0 - 1, rect_b1 + 1]
    )
    rect_factor = beta * (
        rect_k * pt.log1p(-eps) - gamma * rect_log_x + (set_n[rect_set] - rect_k) * log_noise
    ) - shift[rect_set]
    w_rect = pt.maximum(rect_g, 0.0) * pt.exp(rect_factor)
    is_total = pt.lt(rect_cell, 0)
    w_int_sets = _scatter_sum(n_sets, rect_set, w_rect * is_total)
    w_int_cells = _scatter_sum(n_cells, pt.maximum(rect_cell, 0), w_rect * (1 - is_total))
    size_prior = pt.exp(log_size_prior)
    empty_scale = pt.exp(beta * set_n * log_noise - shift)  # (sets,): shared likelihood of empty intervals
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)

    z = w_rules.sum(axis=1) + w_aug.sum(axis=1) + w_int_sets + w_empty  # (sets,)
    numer = (
        ((w_rules + w_aug)[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, chit_cell, w_cls[chit_cls])
        + w_int_cells
        + empty_scale[cell_set] * (cell_empty_size_count * size_prior[None, :]).sum(axis=1)
    )
    in_concept = numer / z[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}

    rects = []  # (set, cell or -1, k, sum log x, a0, a1, b0, b1)
    empty_size_count, cell_empty_size_count = [], []
    set_classes = []
    for i, st in enumerate(unique):
        u = sorted(set(st))
        cnt = [st.count(x) for x in u]
        edges = [0] + u + [101]
        classes = []
        for a in range(len(u)):
            for b in range(a, len(u)):
                k = sum(cnt[a:b + 1])
                lx = sum(c * np.log(x) for c, x in zip(cnt[a:b + 1], u[a:b + 1]))
                cl = (k, lx, edges[a] + 1, edges[a + 1], edges[b + 1], edges[b + 2] - 1)
                classes.append(cl)
                rects.append((i, -1) + cl)
        set_classes.append(classes)
        gaps = [edges[j + 1] - edges[j] - 1 for j in range(len(edges) - 1)]
        esc = np.zeros(100)
        for gl in gaps:
            for w in range(1, gl + 1):
                esc[w - 1] += gl - w + 1
        empty_size_count.append(esc)
    for c, (i, t) in enumerate(cells):
        for k, lx, a0, a1, b0, b1 in set_classes[i]:
            a1t, b0t = min(a1, t), max(b0, t)
            if a0 <= a1t and b0t <= b1:
                rects.append((i, c, k, lx, a0, a1t, b0t, b1))
        st = unique[i]
        esc = np.zeros(100)
        if t not in st:
            lo = max([x for x in st if x < t], default=0) + 1
            hi = min([x for x in st if x > t], default=101) - 1
            for a in range(lo, t + 1):
                for b in range(t, hi + 1):
                    esc[b - a] += 1
        cell_empty_size_count.append(esc)
    rc = np.array(rects, dtype="float64")

    cls_rows = []  # (set, rule, h, k, m, k_rule, extra size)
    cls_extra = []  # per concept: set of non-rule numbers its neighbourhood adds
    set_cls = []  # per set: concept indices
    for i, st in enumerate(unique):
        ex_count = np.bincount(np.array(st), minlength=101)[1:]
        mine = []
        for r in range(N_RULES):
            in_rule = RULE_MEMBERSHIP[r] > 0
            k_rule = ex_count[in_rule].sum()
            centres = sorted({x for x in st if not in_rule[x - 1]})
            for x in centres:
                for h in range(H_MAX + 1):
                    extra = np.arange(max(1, x - h), min(100, x + h) + 1)
                    extra = extra[~in_rule[extra - 1]]
                    mine.append(len(cls_rows))
                    cls_rows.append((i, r, h, k_rule + ex_count[extra - 1].sum(), len(centres), k_rule, len(extra)))
                    cls_extra.append(set(extra.tolist()))
        set_cls.append(mine)
    chit_cell, chit_cls = [], []
    for c, (i, t) in enumerate(cells):
        for j in set_cls[i]:
            if t in cls_extra[j]:
                chit_cell.append(c)
                chit_cls.append(j)
    cls = np.array(cls_rows, dtype="float64")

    return {
        "cls_set": cls[:, 0].astype("int64"),
        "cls_rule": cls[:, 1].astype("int64"),
        "cls_h": cls[:, 2].astype("int64"),
        "cls_k": cls[:, 3],
        "cls_log_m": np.log(cls[:, 4]),
        "cls_n_out": np.array([len(unique[int(i)]) for i in cls[:, 0]], dtype="float64") - cls[:, 3],
        "cls_krule": cls[:, 5],
        "cls_esize": cls[:, 6],
        "chit_cell": np.array(chit_cell, dtype="int64"),
        "chit_cls": np.array(chit_cls, dtype="int64"),
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "rect_set": rc[:, 0].astype("int64"),
        "rect_cell": rc[:, 1].astype("int64"),
        "rect_k": rc[:, 2].astype("int64"),
        "rect_log_x": rc[:, 3],
        "rect_a0": rc[:, 4].astype("int64"),
        "rect_a1": rc[:, 5].astype("int64"),
        "rect_b0": rc[:, 6].astype("int64"),
        "rect_b1": rc[:, 7].astype("int64"),
        "empty_size_count": np.stack(empty_size_count),
        "cell_empty_size_count": np.stack(cell_empty_size_count),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
