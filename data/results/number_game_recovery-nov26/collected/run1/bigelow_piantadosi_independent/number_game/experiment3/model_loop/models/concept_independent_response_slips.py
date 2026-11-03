"""Concept-independent response slips on top of anchored-neighbourhood side-channel concept learning.

Refinement of `anchored_neighbourhood_side_channel`: the same 45 rules (final-digit
rules boosted by kappa), Erlang-width intervals with compressed-number-line
sampling, tempered size principle (beta), exception rate eps, and "rule plus
exception" concepts whose exception is a neighbourhood [x - h, x + h] around an
off-rule shown example, emitted through a side channel at rate omega. The one
change: on some trials people answer without consulting the inferred concept --
a stray yes at rate `slip_yes`, a stray no at rate `slip_no` -- so
P(yes) = slip_yes + (1 - slip_yes - slip_no) * P(target in concept). These
concept-independent answers pull every set toward a common base rate, so sets
differ less in overall generalisation breadth than pure concept inference
predicts (the critique: the incumbent exaggerates between-set heterogeneity).
The interval part is computed densely per set (set x interval matrix, then a
containment matrix product) instead of sparse scatters, for speed only.
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
# All intervals [a, b] of 1..100.
INT_A = np.array([a for a in NUMBERS for b in NUMBERS if b >= a])
INT_B = np.array([b for a in NUMBERS for b in NUMBERS if b >= a])
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
INT_SIZE_IDX = INT_B - INT_A  # width - 1
N_INT = len(INT_A)
# (intervals, targets): whether interval contains the target.
INT_CONTAINS = ((INT_A[:, None] <= NUMBERS[None, :]) & (INT_B[:, None] >= NUMBERS[None, :])).astype("float64")
H_MAX = 3  # largest exception-neighbourhood radius
H_RADII = np.arange(H_MAX + 1, dtype="float64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)  # intervals of each width


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    # Dense (sets, intervals): examples inside each interval and the sum of their log x.
    int_k = pm.Data("int_k", np.zeros((1, N_INT)))
    int_log_x = pm.Data("int_log_x", np.zeros((1, N_INT)))
    # Cells (set, target): which listed intervals contain the target.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_t = pm.Data("cell_t", np.zeros(1, dtype="int64"))  # target - 1
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
    slip_yes = pm.Beta("slip_yes", 1, 19)  # stray yes without consulting the concept
    slip_no = pm.Beta("slip_no", 1, 19)  # stray no without consulting the concept

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

    # Rule R plus neighbourhood N of radius h around a shown example x not in R, as two channels:
    # rule members w.p. 1 - omega, members of N \ R uniformly w.p. omega.
    log_h_prior = -rho * H_RADII - pm.math.logsumexp(-rho * H_RADII)
    side_rule_in = pt.log((1 - eps) * (1 - omega) / RULE_SIZE + eps / 100)  # (rules,)
    cls_x_in = pt.log((1 - eps) * omega / cls_extra_n + eps / 100)
    k_rule = rule_k[cls_set, cls_rule]
    log_cls = (
        pt.log(lambda_0) + pt.log(alpha) + log_rule_prior[cls_rule] + log_h_prior[cls_h]
        - pt.log(cls_m)
        + beta * (
            k_rule * side_rule_in[cls_rule]
            + cls_e * cls_x_in
            + (set_n[cls_set] - k_rule - cls_e) * log_noise
        )
    )
    w_cls = pt.exp(log_cls - shift[cls_set])
    w_aug = _scatter_sum(n_sets * N_RULES, cls_set * N_RULES + cls_rule, w_cls).reshape((n_sets, N_RULES))

    # Interval members are drawn with weight x^-gamma: P(x | [a, b]) = x^-gamma / W(a, b).
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    log_w = pt.log(cum_w[INT_B] - cum_w[INT_A - 1])  # (intervals,)
    int_lik = (
        int_k * (pt.log1p(-eps) - log_w[None, :])
        - gamma * int_log_x
        + (set_n[:, None] - int_k) * log_noise
    )  # (sets, intervals)
    # Erlang width prior: log P(interval of width s) = log s - s / sigma - log Z.
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z  # (widths,)
    w_int = pt.exp(log_size_prior[INT_SIZE_IDX][None, :] + beta * int_lik - shift[:, None])
    w_int_t = pt.dot(w_int, INT_CONTAINS)  # (sets, targets): interval mass holding each target

    z = w_rules.sum(axis=1) + w_aug.sum(axis=1) + w_int.sum(axis=1)  # (sets,)
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
        + (w_aug[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, chit_cell, w_cls[chit_cls])
        + w_int_t[cell_set, cell_t]
    )
    in_concept = numer / z[cell_set]

    p_judged = slip_yes + (1 - slip_yes - slip_no) * in_concept
    p_yes = pm.Deterministic("p_yes", pt.clip(p_judged[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}

    int_k, int_log_x = [], []
    for st in unique:
        ex = np.array(st)
        inside = (INT_A[:, None] <= ex[None, :]) & (INT_B[:, None] >= ex[None, :])
        int_k.append(inside.sum(axis=1).astype("float64"))
        int_log_x.append((inside * np.log(ex)[None, :]).sum(axis=1))

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
        "int_k": np.stack(int_k),
        "int_log_x": np.stack(int_log_x),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_t": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
