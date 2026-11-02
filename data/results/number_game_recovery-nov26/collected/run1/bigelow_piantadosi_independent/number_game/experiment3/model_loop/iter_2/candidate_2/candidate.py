"""Even-leaning rule prior.

People expect a number-producing program to be "about" even numbers more
readily than about odd ones: a rule's prior grows with the share of its
members that are even (log prior + zeta * (share even - 1/2)), so "even",
"multiples of 4", "powers of 2", "ends in 6" start ahead of "odd", "ends in 7",
"powers of 3" or "primes". Otherwise this is the best model
(`anchored_neighbourhood_side_channel`): 45 rules (final-digit rules boosted by
kappa), Erlang-width intervals with compressed-number-line sampling, tempered
size principle (beta), exception rate eps, and "rule plus anchored
neighbourhood" concepts emitted through a side channel (rate omega). Ranges are
parity-neutral, so the tilt shows wherever the examples leave both parities
open, most clearly as an even-target bias in mixed-parity sets.

Intervals are not enumerated per set: an interval's weight factors into a
set-independent table over (a, b) for each count of examples inside times a
constant per region of intervals holding the same examples, so region sums
(and those holding a target) come from 2D prefix sums.
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
# Share of each rule's members that are even, centred: +1/2 for "even", -1/2 for "odd".
RULE_EVEN = RULE_MEMBERSHIP @ (NUMBERS % 2 == 0).astype("float64") / RULE_SIZE - 0.5
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
GRID_A, GRID_B = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
TRI = (GRID_B >= GRID_A).astype("float64")
GRID_W = np.clip(GRID_B - GRID_A, 0, 99)  # width - 1
K_RANGE = np.arange(5, dtype="float64")
MAX_EX = 4
H_MAX = 3
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
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)
    eps = pm.Beta("eps", 2, 5)
    gamma = pm.HalfNormal("gamma", sigma=1.0)
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)
    zeta = pm.Normal("zeta", mu=0.0, sigma=2.0)  # log prior advantage of an all-even over an all-odd rule
    alpha = pm.Beta("alpha", 1, 4)
    rho = pm.LogNormal("rho", mu=0.0, sigma=1.0)
    omega = pm.Beta("omega", 2, 6)

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    rule_score = kappa * RULE_SALIENT + zeta * RULE_EVEN
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
    log_width = pt.log(SIZES) - SIZES / sigma
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

    reg_const = pt.exp(
        g_max[reg_k]
        + beta * (reg_k * pt.log1p(-eps) - gamma * reg_log_x + (set_n[reg_set] - reg_k) * log_noise)
        - shift[reg_set]
    )
    w_reg = rect(reg_corner) * reg_const
    covered = _scatter_sum(n_cells, hit_cell, rect(hit_corner) * reg_const[hit_reg])

    z = w_rules.sum(axis=1) + w_aug.sum(axis=1) + _scatter_sum(n_sets, reg_set, w_reg)
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
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
