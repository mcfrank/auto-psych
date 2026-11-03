"""Range-plus-exception programs whose ranges follow a gamma-shaped width prior.

Refinement of `range_plus_exception_side_channel`: people do the same tempered,
exception-tolerant Bayesian concept learning (45 rules with salient final-digit
rules, ranges sampled along a compressed number line, rule-plus-neighbourhood
exceptions anchored on an off-pattern example and emitted through a side
channel) and also entertain range-plus-exception programs ("numbers from 20 to
35, and now and then something around 87"), whose range cedes a share alpha of
its prior to variants adding a neighbourhood [x - h, x + h] (h = 0..3,
P(h) ~ exp(-rho h)) around an example outside it, emitted at rate omega.

The one change (grafted from `gamma_shaped_range_width_prior`): the prior on a
range's width is not the fixed Erlang shape (width x exp(-width / sigma)) but
a gamma shape whose shape and mean are fitted, applied to plain ranges and to
the main channel of range-plus-exception programs alike. If tight ranges are
plausible, a clustered pair or a lone example yields a tight cluster range,
and the outlier's neighbourhood can be explained by a range-plus-exception
program with a tight main range, so people fill in a narrow stretch around the
cluster, endorse the outlier's close neighbours, and reject the gap between.

Intervals are not enumerated per set: an interval's weight factors into a
set-independent table over (a, b) for each count of examples inside times a
constant per region of intervals holding the same examples, so region sums
(and those holding a target) come from 2D prefix sums; a range-plus-exception
variant is the same region clipped to lie clear of its neighbourhood.
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
RULE_SALIENT = np.array([n in ("odd", "even") or n.startswith("ends in") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
GRID_A, GRID_B = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
TRI = (GRID_B >= GRID_A).astype("float64")
GRID_W = np.clip(GRID_B - GRID_A, 0, 99)  # width - 1
K_RANGE = np.arange(5, dtype="float64")
MAX_EX = 4
H_MAX = 3  # largest exception-neighbourhood radius
H_RADII = np.arange(H_MAX + 1, dtype="float64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))
    set_n = pm.Data("set_n", np.ones(1))
    reg_set = pm.Data("reg_set", np.zeros(1, dtype="int64"))
    reg_k = pm.Data("reg_k", np.zeros(1, dtype="int64"))  # examples inside the region's intervals
    reg_log_x = pm.Data("reg_log_x", np.zeros(1))  # sum of log x over them
    reg_corner = pm.Data("reg_corner", np.zeros((1, 4), dtype="int64"))
    hit_cell = pm.Data("hit_cell", np.zeros(1, dtype="int64"))
    hit_reg = pm.Data("hit_reg", np.zeros(1, dtype="int64"))
    hit_corner = pm.Data("hit_corner", np.zeros((1, 4), dtype="int64"))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    cls_set = pm.Data("cls_set", np.zeros(1, dtype="int64"))
    cls_rule = pm.Data("cls_rule", np.zeros(1, dtype="int64"))
    cls_h = pm.Data("cls_h", np.zeros(1, dtype="int64"))
    cls_extra_n = pm.Data("cls_extra_n", np.ones(1))
    cls_e = pm.Data("cls_e", np.zeros(1))
    cls_m = pm.Data("cls_m", np.ones(1))
    chit_cell = pm.Data("chit_cell", np.zeros(1, dtype="int64"))
    chit_cls = pm.Data("chit_cls", np.zeros(1, dtype="int64"))
    # Range-plus-neighbourhood variants: a region clipped clear of the neighbourhood around an off-range example.
    rx_reg = pm.Data("rx_reg", np.zeros(1, dtype="int64"))  # region whose held examples the range holds
    rx_h = pm.Data("rx_h", np.zeros(1, dtype="int64"))
    rx_nb_n = pm.Data("rx_nb_n", np.ones(1))  # numbers in the neighbourhood
    rx_e = pm.Data("rx_e", np.zeros(1))  # examples in the neighbourhood
    rx_m = pm.Data("rx_m", np.ones(1))  # off-range examples the exception may be anchored on
    rx_corner = pm.Data("rx_corner", np.zeros((1, 4), dtype="int64"))
    # Cells whose target lies in the variant's neighbourhood (whole variant) or in its range (clipped further).
    rxn_cell = pm.Data("rxn_cell", np.zeros(1, dtype="int64"))
    rxn_rx = pm.Data("rxn_rx", np.zeros(1, dtype="int64"))
    rxr_cell = pm.Data("rxr_cell", np.zeros(1, dtype="int64"))
    rxr_rx = pm.Data("rxr_rx", np.zeros(1, dtype="int64"))
    rxr_corner = pm.Data("rxr_corner", np.zeros((1, 4), dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)
    eps = pm.Beta("eps", 2, 5)
    gamma = pm.HalfNormal("gamma", sigma=1.0)
    mean_width = pm.LogNormal("mean_width", mu=np.log(40.0), sigma=1.0)  # typical range width
    width_shape = pm.LogNormal("width_shape", mu=np.log(2.0), sigma=0.75)  # gamma shape of the width prior (2 = Erlang)
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)
    alpha = pm.Beta("alpha", 1, 4)
    rho = pm.LogNormal("rho", mu=0.0, sigma=1.0)
    omega = pm.Beta("omega", 2, 6)

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    rule_score = kappa * RULE_SALIENT
    log_rule_prior = rule_score - pm.math.logsumexp(rule_score)
    has_off = pt.gt(set_n[:, None] - rule_k, 0.5)
    log_rules = pt.log(lambda_0) + pt.log1p(-alpha * has_off) + log_rule_prior[None, :] + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)
    w_rules = pt.exp(log_rules - shift[:, None])

    # Rule plus anchored neighbourhood, two channels (rule w.p. 1 - omega, neighbourhood w.p. omega).
    log_h_prior = -rho * H_RADII - pm.math.logsumexp(-rho * H_RADII)
    side_rule_in = pt.log((1 - eps) * (1 - omega) / RULE_SIZE + eps / 100)
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

    # Intervals: members drawn with weight x^-gamma, P(x | [a, b]) = x^-gamma / W(a, b).
    # Weight of [a, b] = G_k(a, b) * C(set, region), G_k = P(width) W^(-beta k).
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    log_w_int = pt.log(pt.maximum(cum_w[GRID_B] - cum_w[GRID_A - 1], 1e-30))
    # Gamma width prior, shared by plain ranges and the main channel of range-plus-exception programs.
    log_width = (width_shape - 1) * pt.log(SIZES) - SIZES * width_shape / mean_width
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z
    log_g = log_size_prior[GRID_W][None, :, :] - beta * K_RANGE[:, None, None] * log_w_int[None, :, :]
    log_g = pt.where(TRI[None] > 0, log_g, -1e4)
    g_max = pt.max(log_g, axis=(1, 2))
    g = pt.exp(log_g - g_max[:, None, None])
    prefix = pt.zeros((MAX_EX + 1, 101, 101))
    prefix = pt.set_subtensor(prefix[:, 1:, 1:], pt.cumsum(pt.cumsum(g, axis=1), axis=2)).reshape((-1,))

    def rect(corner):
        sums = prefix[corner[:, 0]] - prefix[corner[:, 1]] - prefix[corner[:, 2]] + prefix[corner[:, 3]]
        return pt.maximum(sums, 0.0)

    # A range holding some but not all examples cedes alpha to its range-plus-exception variants.
    reg_cedes = pt.gt(reg_k, 0) * pt.lt(reg_k, set_n[reg_set])
    reg_const = pt.exp(
        g_max[reg_k]
        + pt.log1p(-alpha * reg_cedes)
        + beta * (reg_k * pt.log1p(-eps) - gamma * reg_log_x + (set_n[reg_set] - reg_k) * log_noise)
        - shift[reg_set]
    )
    w_reg = rect(reg_corner) * reg_const
    covered = _scatter_sum(n_cells, hit_cell, rect(hit_corner) * reg_const[hit_reg])

    # Range plus neighbourhood, two channels: range members w.p. 1 - omega, neighbourhood w.p. omega.
    rx_k = reg_k[rx_reg]
    rx_set = reg_set[rx_reg]
    rx_const = pt.exp(
        g_max[rx_k]
        + pt.log(alpha) + log_h_prior[rx_h] - pt.log(rx_m)
        + beta * (
            rx_k * (pt.log1p(-eps) + pt.log1p(-omega)) - gamma * reg_log_x[rx_reg]
            + rx_e * pt.log((1 - eps) * omega / rx_nb_n + eps / 100)
            + (set_n[rx_set] - rx_k - rx_e) * log_noise
        )
        - shift[rx_set]
    )
    w_rx = rect(rx_corner) * rx_const
    covered = (
        covered
        + _scatter_sum(n_cells, rxn_cell, w_rx[rxn_rx])
        + _scatter_sum(n_cells, rxr_cell, rect(rxr_corner) * rx_const[rxr_rx])
    )

    z = w_rules.sum(axis=1) + w_aug.sum(axis=1) + _scatter_sum(n_sets, reg_set, w_reg) + _scatter_sum(n_sets, rx_set, w_rx)
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
        + (w_aug[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, chit_cell, w_cls[chit_cls])
        + covered
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

    def corners(k, a0, a1, b0, b1):  # flat prefix indices for the sum over a0..a1 x b0..b1
        base = k * 101 * 101
        return [base + a1 * 101 + b1, base + (a0 - 1) * 101 + b1, base + a1 * 101 + b0 - 1, base + (a0 - 1) * 101 + b0 - 1]

    reg_rows, reg_corner, set_regs = [], [], []
    for i, st in enumerate(unique):
        e = [0] + sorted(set(st)) + [101]
        mine = []
        for gi in range(len(e) - 1):
            for gj in range(gi, len(e) - 1):
                a0, a1 = e[gi] + 1, min(e[gi + 1], 100)
                b0, b1 = max(e[gj], 1), e[gj + 1] - 1
                if a0 > a1 or b0 > b1:
                    continue
                held = [x for x in st if a1 <= x <= b0] if gj > gi else []
                mine.append((len(reg_rows), a0, a1, b0, b1, len(held)))
                reg_rows.append((i, len(held), float(sum(LOG_NUMBERS[x - 1] for x in held))))
                reg_corner.append(corners(len(held), a0, a1, b0, b1))
        set_regs.append(mine)
    hit_cell, hit_reg, hit_corner = [], [], []
    for c, (i, t) in enumerate(cells):
        for r, a0, a1, b0, b1, k in set_regs[i]:
            ca1, cb0 = min(a1, t), max(b0, t)
            if a0 <= ca1 and cb0 <= b1:
                hit_cell.append(c)
                hit_reg.append(r)
                hit_corner.append(corners(k, a0, ca1, cb0, b1))
    reg = np.array(reg_rows, dtype="float64")

    rx_rows, rx_corner, set_rx = [], [], [[] for _ in unique]
    for i, st in enumerate(unique):
        n = len(st)
        for r, a0, a1, b0, b1, k in set_regs[i]:
            if k == 0 or k == n:
                continue
            held = {x for x in st if a1 <= x <= b0}
            anchors = sorted(set(st) - held)
            for x in anchors:
                for h in range(H_MAX + 1):
                    lo, hi = max(1, x - h), min(100, x + h)
                    na0, nb1 = a0, b1
                    if x < a1:  # anchor below the range: the range starts above the neighbourhood
                        na0 = max(a0, hi + 1)
                    else:  # anchor above the range: the range ends below the neighbourhood
                        nb1 = min(b1, lo - 1)
                    if na0 > a1 or b0 > nb1:
                        continue
                    e_n = sum(1 for y in st if lo <= y <= hi)
                    set_rx[i].append((len(rx_rows), na0, a1, b0, nb1, k, lo, hi))
                    rx_rows.append((r, h, hi - lo + 1, e_n, len(anchors)))
                    rx_corner.append(corners(k, na0, a1, b0, nb1))
    rxn_cell, rxn_rx, rxr_cell, rxr_rx, rxr_corner = [], [], [], [], []
    for c, (i, t) in enumerate(cells):
        for j, a0, a1, b0, b1, k, lo, hi in set_rx[i]:
            if lo <= t <= hi:
                rxn_cell.append(c)
                rxn_rx.append(j)
                continue
            ca1, cb0 = min(a1, t), max(b0, t)
            if a0 <= ca1 and cb0 <= b1:
                rxr_cell.append(c)
                rxr_rx.append(j)
                rxr_corner.append(corners(k, a0, ca1, cb0, b1))
    rx = np.array(rx_rows, dtype="float64").reshape(-1, 5)

    cls_rows, cls_extra, set_cls = [], [], []
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
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "reg_set": reg[:, 0].astype("int64"),
        "reg_k": reg[:, 1].astype("int64"),
        "reg_log_x": reg[:, 2],
        "reg_corner": np.array(reg_corner, dtype="int64"),
        "hit_cell": np.array(hit_cell, dtype="int64"),
        "hit_reg": np.array(hit_reg, dtype="int64"),
        "hit_corner": np.array(hit_corner, dtype="int64"),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "cls_set": cls[:, 0].astype("int64"),
        "cls_rule": cls[:, 1].astype("int64"),
        "cls_h": cls[:, 2].astype("int64"),
        "cls_extra_n": cls[:, 3],
        "cls_e": cls[:, 4],
        "cls_m": cls[:, 5],
        "chit_cell": np.array(chit_cell, dtype="int64"),
        "chit_cls": np.array(chit_cls, dtype="int64"),
        "rx_reg": rx[:, 0].astype("int64"),
        "rx_h": rx[:, 1].astype("int64"),
        "rx_nb_n": rx[:, 2],
        "rx_e": rx[:, 3],
        "rx_m": rx[:, 4],
        "rx_corner": np.array(rx_corner, dtype="int64").reshape(-1, 4),
        "rxn_cell": np.array(rxn_cell, dtype="int64"),
        "rxn_rx": np.array(rxn_rx, dtype="int64"),
        "rxr_cell": np.array(rxr_cell, dtype="int64"),
        "rxr_rx": np.array(rxr_rx, dtype="int64"),
        "rxr_corner": np.array(rxr_corner, dtype="int64").reshape(-1, 4),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
