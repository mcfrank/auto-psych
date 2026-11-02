"""Pedagogical (helpful-teacher) sampling of the examples.

People assume the examples were picked by a helpful teacher who wanted them to
guess the program, not drawn at random from it: a teacher of concept h shows
number x in proportion to how strongly x alone would point a naive learner to
h, P_teach(x | h) ~ L0(h | x)^tau over x in h. With L0 a fixed literal learner
(half the prior on the 45 rules, half on all intervals, strong sampling),
L0(h | x) = P(h) / |h| / D(x), where D(x) is how ambiguous x is (how many small
concepts contain it), so within any concept the teacher picks x with weight
D(x)^-tau: ambiguous "special" numbers (16, 64, 81) are expected only when they
are needed. The learner then does the same tempered, exception-tolerant
Bayesian concept learning as the best model -- 45 rules (final-digit rules
salient), Erlang-width intervals, rule-plus-neighbourhood exceptions anchored
on an off-rule example and emitted through a side channel -- but with this
teacher likelihood for every concept in place of random (number-line
compressed) sampling. tau is the strength of the pedagogical assumption.

Intervals are not enumerated per set: an interval's weight factors into a
set-independent table over (a, b) for each count of examples inside and a
constant per region of intervals holding the same examples, so the sums over
regions (and over those holding a target) come from 2D prefix sums.
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
INT_A = np.array([a for a in NUMBERS for b in NUMBERS if b >= a])
INT_B = np.array([b for a in NUMBERS for b in NUMBERS if b >= a])
INT_SIZE_IDX = INT_B - INT_A  # width - 1
# Dense (a, b) grid of intervals: upper triangle valid.
GRID_A, GRID_B = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
TRI = (GRID_B >= GRID_A).astype("float64")
GRID_W = np.clip(GRID_B - GRID_A, 0, 99)
K_RANGE = np.arange(5, dtype="float64")
H_MAX = 3
H_RADII = np.arange(H_MAX + 1, dtype="float64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)
MAX_EX = 4


def _ambiguity():
    """log D(x): a literal learner's total (prior x likelihood) over concepts holding x."""
    d = 0.5 / N_RULES * (RULE_MEMBERSHIP / RULE_SIZE[:, None]).sum(axis=0)
    w = 0.5 / len(INT_A) / (INT_SIZE_IDX + 1.0)
    diff = np.zeros(102)
    np.add.at(diff, INT_A, w)
    np.add.at(diff, INT_B + 1, -w)
    d = d + np.cumsum(diff)[1:101]
    return np.log(d)


LOG_D = _ambiguity()


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    ex_num = pm.Data("ex_num", np.ones((1, MAX_EX), dtype="int64"))  # (sets, 4) example numbers, padded
    ex_in_rule = pm.Data("ex_in_rule", np.zeros((1, MAX_EX, N_RULES)))  # real example inside rule
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))
    set_n = pm.Data("set_n", np.ones(1))
    # Interval regions: per set, the intervals holding the same examples form a rectangle of (a, b).
    reg_set = pm.Data("reg_set", np.zeros(1, dtype="int64"))
    reg_k = pm.Data("reg_k", np.zeros(1, dtype="int64"))  # examples inside
    reg_log_d = pm.Data("reg_log_d", np.zeros(1))  # sum of log D(x) over them
    reg_corner = pm.Data("reg_corner", np.zeros((1, 4), dtype="int64"))  # prefix-sum corners (++, -+, +-, --)
    # The same, clipped to intervals holding the cell's target.
    hit_cell = pm.Data("hit_cell", np.zeros(1, dtype="int64"))
    hit_reg = pm.Data("hit_reg", np.zeros(1, dtype="int64"))
    hit_corner = pm.Data("hit_corner", np.zeros((1, 4), dtype="int64"))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    # Distinct exception regions (off-rule numbers of [x - h, x + h]) and the concepts using them.
    nb_member = pm.Data("nb_member", np.ones((1, 100)))
    cls_set = pm.Data("cls_set", np.zeros(1, dtype="int64"))
    cls_rule = pm.Data("cls_rule", np.zeros(1, dtype="int64"))
    cls_h = pm.Data("cls_h", np.zeros(1, dtype="int64"))
    cls_nb = pm.Data("cls_nb", np.zeros(1, dtype="int64"))
    cls_e = pm.Data("cls_e", np.zeros(1))
    cls_m = pm.Data("cls_m", np.ones(1))
    ce_cls = pm.Data("ce_cls", np.zeros(1, dtype="int64"))  # (concept, example in its exception region)
    ce_num = pm.Data("ce_num", np.ones(1, dtype="int64"))
    chit_cell = pm.Data("chit_cell", np.zeros(1, dtype="int64"))
    chit_cls = pm.Data("chit_cls", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)
    eps = pm.Beta("eps", 2, 5)
    tau = pm.LogNormal("tau", mu=-1.0, sigma=0.75)  # strength of the helpful-teacher assumption
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)
    alpha = pm.Beta("alpha", 1, 4)
    rho = pm.LogNormal("rho", mu=0.0, sigma=1.0)
    omega = pm.Beta("omega", 2, 6)

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    n_cls = cls_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    # Teacher weight of each number: D(x)^-tau (centred for stability).
    log_tw = -tau * (pt.as_tensor(LOG_D) - LOG_D.mean())  # (100,)
    tw = pt.exp(log_tw)
    log_w_rule = pt.log(pt.dot(pt.as_tensor(RULE_MEMBERSHIP), tw))  # (rules,)
    ex_log_tw = log_tw[ex_num - 1]  # (sets, 4)

    def rule_ll(log_scale):
        inside = pt.logaddexp(log_scale + ex_log_tw[:, :, None] - log_w_rule[None, None, :], log_noise)
        return (ex_in_rule * inside).sum(axis=1)  # (sets, rules)

    rule_log_lik = rule_ll(pt.log1p(-eps)) + (set_n[:, None] - rule_k) * log_noise
    side_in_ll = rule_ll(pt.log1p(-eps) + pt.log1p(-omega))  # in-rule examples only
    log_rule_prior = kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    has_off = pt.gt(set_n[:, None] - rule_k, 0.5)
    log_rules = pt.log(lambda_0) + pt.log1p(-alpha * has_off) + log_rule_prior[None, :] + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)
    w_rules = pt.exp(log_rules - shift[:, None])

    # Rule plus anchored neighbourhood, two channels; the teacher's weights apply in each channel.
    log_h_prior = -rho * H_RADII - pm.math.logsumexp(-rho * H_RADII)
    log_w_nb = pt.log(pt.dot(nb_member, tw))  # (regions,)
    x_ll = pt.logaddexp(
        pt.log1p(-eps) + pt.log(omega) + log_tw[ce_num - 1] - log_w_nb[cls_nb[ce_cls]], log_noise
    )
    k_rule = rule_k[cls_set, cls_rule]
    log_cls = (
        pt.log(lambda_0) + pt.log(alpha) + log_rule_prior[cls_rule] + log_h_prior[cls_h]
        - pt.log(cls_m)
        + beta * (
            side_in_ll[cls_set, cls_rule]
            + _scatter_sum(n_cls, ce_cls, x_ll)
            + (set_n[cls_set] - k_rule - cls_e) * log_noise
        )
    )
    w_cls = pt.exp(log_cls - shift[cls_set])
    w_aug = _scatter_sum(n_sets * N_RULES, cls_set * N_RULES + cls_rule, w_cls).reshape((n_sets, N_RULES))

    # Intervals: teacher shows x in [a, b] with weight D(x)^-tau / W(a, b), W the sum over [a, b].
    # Weight of [a, b] for a set = G_k(a, b) * C(set, region), with G_k(a, b) = P(width) W(a, b)^(-beta k)
    # set-independent; region sums come from 2D prefix sums of G_k.
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(tw)])
    log_w_int = pt.log(pt.maximum(cum_w[GRID_B] - cum_w[GRID_A - 1], 1e-30))  # (100, 100)
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z
    log_g = log_size_prior[GRID_W][None, :, :] - beta * K_RANGE[:, None, None] * log_w_int[None, :, :]
    log_g = pt.where(TRI[None] > 0, log_g, -1e4)
    g_max = pt.max(log_g, axis=(1, 2))  # (5,)
    g = pt.exp(log_g - g_max[:, None, None])
    prefix = pt.zeros((MAX_EX + 1, 101, 101))
    prefix = pt.set_subtensor(prefix[:, 1:, 1:], pt.cumsum(pt.cumsum(g, axis=1), axis=2)).reshape((-1,))

    def rect(corner):
        sums = prefix[corner[:, 0]] - prefix[corner[:, 1]] - prefix[corner[:, 2]] + prefix[corner[:, 3]]
        return pt.maximum(sums, 0.0)

    reg_const = pt.exp(
        g_max[reg_k]
        + beta * (reg_k * pt.log1p(-eps) - tau * (reg_log_d - reg_k * LOG_D.mean()) + (set_n[reg_set] - reg_k) * log_noise)
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

    ex_num = np.ones((len(unique), MAX_EX), dtype="int64")
    ex_mask = np.zeros((len(unique), MAX_EX))
    for i, st in enumerate(unique):
        ex_num[i, : len(st)] = st
        ex_mask[i, : len(st)] = 1.0

    def corners(k, a0, a1, b0, b1):  # flat prefix indices for the sum over a0..a1 x b0..b1
        base = k * 101 * 101
        return [base + a1 * 101 + b1, base + (a0 - 1) * 101 + b1, base + a1 * 101 + b0 - 1, base + (a0 - 1) * 101 + b0 - 1]

    reg_rows, reg_corner, set_regs = [], [], []
    for i, st in enumerate(unique):
        e = [0] + sorted(set(st)) + [101]
        mine = []
        for gi in range(len(e) - 1):
            for gj in range(gi, len(e) - 1):
                a0, a1 = e[gi] + 1, e[gi + 1]
                b0, b1 = max(e[gj], 1), e[gj + 1] - 1
                a1 = min(a1, 100)
                if a0 > a1 or b0 > b1:
                    continue
                held = [x for x in st if a1 <= x <= b0] if gj > gi else []
                mine.append((len(reg_rows), a0, a1, b0, b1, len(held)))
                reg_rows.append((i, len(held), float(sum(LOG_D[x - 1] for x in held))))
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
    ex_in_rule = RULE_MEMBERSHIP[:, ex_num - 1].transpose(1, 2, 0) * ex_mask[:, :, None]

    nb_keys, nb_rows = {}, []
    cls_rows, cls_nb, ce_cls, ce_num, cls_sets, set_cls = [], [], [], [], [], []
    for i, st in enumerate(unique):
        mine = []
        for r in range(N_RULES):
            in_rule = RULE_MEMBERSHIP[r] > 0
            centres = sorted({x for x in st if not in_rule[x - 1]})
            for x in centres:
                for h in range(H_MAX + 1):
                    extra = np.arange(max(1, x - h), min(100, x + h) + 1)
                    extra = extra[~in_rule[extra - 1]]
                    key = tuple(extra.tolist())
                    if key not in nb_keys:
                        nb_keys[key] = len(nb_rows)
                        nb_rows.append(np.isin(NUMBERS, extra).astype("float64"))
                    c = len(cls_rows)
                    n_in = 0
                    for y in st:
                        if y in key:
                            ce_cls.append(c)
                            ce_num.append(y)
                            n_in += 1
                    mine.append(c)
                    cls_rows.append((i, r, h, n_in, len(centres)))
                    cls_nb.append(nb_keys[key])
                    cls_sets.append(set(key))
        set_cls.append(mine)
    chit_cell, chit_cls = [], []
    for c, (i, t) in enumerate(cells):
        for j in set_cls[i]:
            if t in cls_sets[j]:
                chit_cell.append(c)
                chit_cls.append(j)
    cls = np.array(cls_rows, dtype="float64")

    return {
        "ex_num": ex_num,
        "ex_in_rule": ex_in_rule,
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "reg_set": reg[:, 0].astype("int64"),
        "reg_k": reg[:, 1].astype("int64"),
        "reg_log_d": reg[:, 2],
        "reg_corner": np.array(reg_corner, dtype="int64"),
        "hit_cell": np.array(hit_cell, dtype="int64"),
        "hit_reg": np.array(hit_reg, dtype="int64"),
        "hit_corner": np.array(hit_corner, dtype="int64"),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "nb_member": np.stack(nb_rows),
        "cls_set": cls[:, 0].astype("int64"),
        "cls_rule": cls[:, 1].astype("int64"),
        "cls_h": cls[:, 2].astype("int64"),
        "cls_nb": np.array(cls_nb, dtype="int64"),
        "cls_e": cls[:, 3],
        "cls_m": cls[:, 4],
        "ce_cls": np.array(ce_cls, dtype="int64"),
        "ce_num": np.array(ce_num, dtype="int64"),
        "chit_cell": np.array(chit_cell, dtype="int64"),
        "chit_cls": np.array(chit_cls, dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
