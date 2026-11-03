"""Diminishing per-example evidence in the anchored side-channel learner.

Refinement of `anchored_neighbourhood_side_channel`: people do the same
tempered, exception-tolerant Bayesian concept learning (45 rules with salient
final-digit rules, Erlang-width ranges on a compressed number line, and
rule-plus-neighbourhood exceptions anchored on off-rule examples and emitted
through a side channel), but they treat the examples of a set as partly
redundant rather than independent draws: the weight each example carries in
the size principle scales with the number of examples shown as n^-zeta (zeta
fitted; zeta > 0 is redundancy), so the total evidence grows as n^(1 - zeta).
A single example is weighed as in the incumbent; with three or four examples
people tighten their generalisation (from broad rules and wide ranges to
narrow ones) less (or more) than a learner counting every example in full.

Computation: an interval's likelihood depends on the set only through n, k
(examples inside) and their summed log value, and the intervals sharing these
form rectangles of the (start, end) grid, summed by 2-D prefix sums.
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
# All intervals [a, b] of 1..100 as a (start, end) grid; a > b is not an interval.
GRID_A = np.repeat(NUMBERS[:, None], 100, axis=1)
GRID_B = np.repeat(NUMBERS[None, :], 100, axis=0)
VALID = (GRID_B >= GRID_A).astype("float64")
WIDTH_IDX = np.maximum(GRID_B - GRID_A, 0)  # width - 1
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
N_MAX = 4  # most examples in a set
# An interval's likelihood depends on the set only through n (examples shown) and k (examples inside):
# one grid per combo c = (n - 1) * (N_MAX + 1) + k.
COMBO_N = np.repeat(np.arange(1, N_MAX + 1), N_MAX + 1).astype("float64")
COMBO_K = np.tile(np.arange(N_MAX + 1), N_MAX).astype("float64")
H_MAX = 3  # largest exception-neighbourhood radius
H_RADII = np.arange(H_MAX + 1, dtype="float64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)  # intervals of each width


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    # Intervals holding >= 1 example fall into rectangles of the (start, end) grid, one per run of
    # consecutive examples they hold; each rectangle shares k and the sum of log x of those examples.
    grp_set = pm.Data("grp_set", np.zeros(1, dtype="int64"))
    grp_combo = pm.Data("grp_combo", np.zeros(1, dtype="int64"))
    grp_k = pm.Data("grp_k", np.zeros(1))
    grp_log_x = pm.Data("grp_log_x", np.zeros(1))
    grp_box = pm.Data("grp_box", np.zeros((1, 4), dtype="int64"))  # a0, a1 + 1, b0, b1 + 1 (0-based)
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 100)))  # (sets, widths): intervals holding no example
    # The same rectangles clipped to intervals holding a cell's target (empty-interval ones included).
    rect_cell = pm.Data("rect_cell", np.zeros(1, dtype="int64"))
    rect_set = pm.Data("rect_set", np.zeros(1, dtype="int64"))
    rect_combo = pm.Data("rect_combo", np.zeros(1, dtype="int64"))
    rect_k = pm.Data("rect_k", np.zeros(1))
    rect_log_x = pm.Data("rect_log_x", np.zeros(1))
    rect_box = pm.Data("rect_box", np.zeros((1, 4), dtype="int64"))
    # Cells (set, target).
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    # Rule-plus-neighbourhood concepts anchored on an off-rule example: one per (set, rule, centre, radius).
    cls_set = pm.Data("cls_set", np.zeros(1, dtype="int64"))
    cls_rule = pm.Data("cls_rule", np.zeros(1, dtype="int64"))
    cls_h = pm.Data("cls_h", np.zeros(1, dtype="int64"))
    cls_extra_n = pm.Data("cls_extra_n", np.ones(1))  # off-rule numbers in the neighbourhood
    cls_e = pm.Data("cls_e", np.zeros(1))  # examples inside the neighbourhood but outside the rule
    cls_m = pm.Data("cls_m", np.ones(1))  # off-rule examples the rule's exception may be anchored on
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
    omega = pm.Beta("omega", 2, 6)  # rate at which the program emits its exception rather than its rule
    zeta = pm.Normal("zeta", mu=0.0, sigma=0.5)  # how fast per-example evidence shrinks with set size
    beta_n = beta * pt.exp(-zeta * np.log(np.arange(1.0, N_MAX + 1)))  # evidence weight per example, by set size
    beta_set = beta * pt.exp(-zeta * pt.log(set_n))  # (sets,)

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)  # (rules,)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rule_prior = kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    # A rule cedes the share alpha to anchored variants only if some example lies outside it.
    has_off = pt.gt(set_n[:, None] - rule_k, 0.5)
    log_rules = pt.log(lambda_0) + pt.log1p(-alpha * has_off) + log_rule_prior[None, :] + beta_set[:, None] * rule_log_lik
    shift = pt.max(log_rules, axis=1)  # per-set stabiliser
    w_rules = pt.exp(log_rules - shift[:, None])

    # Rule R plus neighbourhood N of radius h around a shown example x not in R, as two channels:
    # rule members w.p. 1 - omega, members of N \ R uniformly w.p. omega.
    log_h_prior = -rho * H_RADII - pm.math.logsumexp(-rho * H_RADII)
    side_rule_in = pt.log((1 - eps) * (1 - omega) / RULE_SIZE + eps / 100)  # (rules,)
    cls_x_in = pt.log((1 - eps) * omega / cls_extra_n + eps / 100)
    k_rule = rule_k[cls_set, cls_rule]
    log_cls = (
        pt.log(lambda_0) + pt.log(alpha) + log_rule_prior[cls_rule] + log_h_prior[cls_h]
        - pt.log(cls_m)
        + beta_set[cls_set] * (
            k_rule * side_rule_in[cls_rule]
            + cls_e * cls_x_in
            + (set_n[cls_set] - k_rule - cls_e) * log_noise
        )
    )
    w_cls = pt.exp(log_cls - shift[cls_set])
    w_aug = _scatter_sum(n_sets * N_RULES, cls_set * N_RULES + cls_rule, w_cls).reshape((n_sets, N_RULES))

    # Interval members are drawn with weight x^-gamma: P(x | [a, b]) = x^-gamma / W(a, b).
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    log_w = pt.log(pt.switch(VALID > 0, cum_w[GRID_B] - cum_w[GRID_A - 1], 1.0))  # (100, 100)
    # Erlang width prior: log P(interval of width s) = log s - s / sigma - log Z.
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z  # (widths,)
    # Per combo (n, k): prior times W(a, b)^(-beta_n k); the rest of the likelihood is per rectangle.
    beta_c = beta_n[(COMBO_N - 1).astype("int64")]
    grid = VALID[None] * pt.exp(log_size_prior[WIDTH_IDX][None] - (beta_c * COMBO_K)[:, None, None] * log_w[None])
    csum = pt.cumsum(pt.cumsum(grid, axis=1), axis=2)
    csum = pt.concatenate([pt.zeros((COMBO_N.size, 1, 100)), csum], axis=1)
    csum = pt.concatenate([pt.zeros((COMBO_N.size, 101, 1)), csum], axis=2)  # (combos, 101, 101) prefix sums

    def box_sum(combo, box):
        a0, a1, b0, b1 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        return csum[combo, a1, b1] - csum[combo, a0, b1] - csum[combo, a1, b0] + csum[combo, a0, b0]

    def box_scale(st, k, log_x):
        return pt.exp(
            beta_set[st] * (k * pt.log1p(-eps) - gamma * log_x + (set_n[st] - k) * log_noise) - shift[st]
        )

    w_grp = box_scale(grp_set, grp_k, grp_log_x) * box_sum(grp_combo, grp_box)
    size_prior = pt.exp(log_size_prior)
    empty_scale = pt.exp(beta_set * set_n * log_noise - shift)  # (sets,): shared likelihood of empty intervals
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)
    w_rect = box_scale(rect_set, rect_k, rect_log_x) * box_sum(rect_combo, rect_box)

    z = w_rules.sum(axis=1) + w_aug.sum(axis=1) + _scatter_sum(n_sets, grp_set, w_grp) + w_empty  # (sets,)
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
        + (w_aug[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, chit_cell, w_cls[chit_cls])
        + _scatter_sum(n_cells, rect_cell, w_rect)
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

    grp, rect, empty_size_count = [], [], []  # grp: (set, combo, k, log x, box); rect: (cell, *grp)
    for i, st in enumerate(unique):
        vals, mult = np.unique(np.array(st), return_counts=True)
        edges = np.concatenate([[0], vals, [101]])  # sentinels
        n = len(st)
        logs = mult * np.log(vals)
        widths = []
        for g in range(len(vals) + 1):  # gap of example-free numbers below vals[g]
            gap = edges[g + 1] - edges[g] - 1
            widths += [w for w in range(gap) for _ in range(gap - w)]
        empty_size_count.append(np.bincount(np.array(widths, dtype="int64"), minlength=100).astype("float64"))
        for lo in range(len(vals)):
            for hi in range(lo, len(vals)):
                k = float(mult[lo:hi + 1].sum())
                box = (edges[lo], edges[lo + 1], edges[hi + 1] - 1, edges[hi + 2] - 1)  # a in [box0, box1), b likewise
                grp.append((i, (n - 1) * (N_MAX + 1) + int(k), k, float(logs[lo:hi + 1].sum()), box))
    for c, (i, t) in enumerate(cells):
        st = unique[i]
        vals, mult = np.unique(np.array(st), return_counts=True)
        edges = np.concatenate([[0], vals, [101]])
        n = len(st)
        logs = mult * np.log(vals)
        if t not in vals:  # intervals holding t but no example
            g = int(np.searchsorted(vals, t))
            rect.append((c, i, (n - 1) * (N_MAX + 1), 0.0, 0.0, (edges[g], t, t - 1, edges[g + 1] - 1)))
        for lo in range(len(vals)):
            for hi in range(lo, len(vals)):
                box = (edges[lo], min(edges[lo + 1], t), max(edges[hi + 1], t) - 1, edges[hi + 2] - 1)
                if box[1] > box[0] and box[3] > box[2]:
                    k = float(mult[lo:hi + 1].sum())
                    rect.append((c, i, (n - 1) * (N_MAX + 1) + int(k), k, float(logs[lo:hi + 1].sum()), box))

    cls_rows = []  # (set, rule, h, off-rule numbers in neighbourhood, examples among them, m)
    cls_extra = []  # per concept: set of non-rule numbers its neighbourhood adds
    set_cls = []  # per set: concept indices
    for i, st in enumerate(unique):
        ex_count = np.bincount(np.array(st), minlength=101)[1:]
        mine = []
        for r in range(N_RULES):
            in_rule = RULE_MEMBERSHIP[r] > 0
            centres = sorted({x for x in st if not in_rule[x - 1]})
            for x in centres:
                for h in range(H_MAX + 1):
                    extra = np.arange(max(1, x - h), min(100, x + h) + 1)
                    extra = extra[~in_rule[extra - 1]]
                    mine.append(len(cls_rows))
                    cls_rows.append((i, r, h, len(extra), ex_count[extra - 1].sum(), len(centres)))
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
        "cls_extra_n": cls[:, 3],
        "cls_e": cls[:, 4],
        "cls_m": cls[:, 5],
        "chit_cell": np.array(chit_cell, dtype="int64"),
        "chit_cls": np.array(chit_cls, dtype="int64"),
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "grp_set": np.array([g[0] for g in grp], dtype="int64"),
        "grp_combo": np.array([g[1] for g in grp], dtype="int64"),
        "grp_k": np.array([g[2] for g in grp], dtype="float64"),
        "grp_log_x": np.array([g[3] for g in grp], dtype="float64"),
        "grp_box": np.array([g[4] for g in grp], dtype="int64"),
        "empty_size_count": np.stack(empty_size_count),
        "rect_cell": np.array([r[0] for r in rect], dtype="int64"),
        "rect_set": np.array([r[1] for r in rect], dtype="int64"),
        "rect_combo": np.array([r[2] for r in rect], dtype="int64"),
        "rect_k": np.array([r[3] for r in rect], dtype="float64"),
        "rect_log_x": np.array([r[4] for r in rect], dtype="float64"),
        "rect_box": np.array([r[5] for r in rect], dtype="int64"),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
