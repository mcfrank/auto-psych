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
H_MAX = 3
H_RADII = np.arange(H_MAX + 1, dtype="float64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)
MAX_EX = 4
MAX_EXTRA = 2 * H_MAX + 1


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
    ex_mask = pm.Data("ex_mask", np.ones((1, MAX_EX)))  # real examples
    ex_in_rule = pm.Data("ex_in_rule", np.zeros((1, MAX_EX, N_RULES)))  # example inside rule
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))
    set_n = pm.Data("set_n", np.ones(1))
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_int = pm.Data("pair_int", np.zeros(1, dtype="int64"))
    pair_k = pm.Data("pair_k", np.zeros(1))
    pair_log_d = pm.Data("pair_log_d", np.zeros(1))  # sum of log D(x) over examples in the interval
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 100)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_empty_size_count = pm.Data("cell_empty_size_count", np.zeros((1, 100)))
    hit_pair = pm.Data("hit_pair", np.zeros(1, dtype="int64"))
    hit_cell = pm.Data("hit_cell", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    cls_set = pm.Data("cls_set", np.zeros(1, dtype="int64"))
    cls_rule = pm.Data("cls_rule", np.zeros(1, dtype="int64"))
    cls_h = pm.Data("cls_h", np.zeros(1, dtype="int64"))
    cls_extra = pm.Data("cls_extra", np.ones((1, MAX_EXTRA), dtype="int64"))  # off-rule numbers in neighbourhood
    cls_extra_mask = pm.Data("cls_extra_mask", np.ones((1, MAX_EXTRA)))
    cls_ex_in = pm.Data("cls_ex_in", np.zeros((1, MAX_EX)))  # set example j lies in the neighbourhood, off-rule
    cls_e = pm.Data("cls_e", np.zeros(1))
    cls_m = pm.Data("cls_m", np.ones(1))
    chit_cell = pm.Data("chit_cell", np.zeros(1, dtype="int64"))
    chit_cls = pm.Data("chit_cls", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)
    eps = pm.Beta("eps", 2, 5)
    tau = pm.LogNormal("tau", mu=0.0, sigma=0.75)  # strength of the helpful-teacher assumption
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)
    alpha = pm.Beta("alpha", 1, 4)
    rho = pm.LogNormal("rho", mu=0.0, sigma=1.0)
    omega = pm.Beta("omega", 2, 6)

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    # Teacher weight of each number: D(x)^-tau (centred for stability).
    log_tw = -tau * (pt.as_tensor(LOG_D) - LOG_D.mean())  # (100,)
    tw = pt.exp(log_tw)
    log_w_rule = pt.log(pt.dot(pt.as_tensor(RULE_MEMBERSHIP), tw))  # (rules,)
    ex_log_tw = log_tw[ex_num - 1]  # (sets, 4)

    def rule_ll(log_scale):
        inside = pt.logaddexp(
            log_scale + ex_log_tw[:, :, None] - log_w_rule[None, None, :], log_noise
        )
        return (ex_mask[:, :, None] * ex_in_rule * inside).sum(axis=1)  # (sets, rules)

    rule_log_lik = rule_ll(pt.log1p(-eps)) + (set_n[:, None] - rule_k) * log_noise
    side_in_ll = rule_ll(pt.log1p(-eps) + pt.log1p(-omega))  # in-rule examples only
    log_rule_prior = kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    has_off = pt.gt(set_n[:, None] - rule_k, 0.5)
    log_rules = pt.log(lambda_0) + pt.log1p(-alpha * has_off) + log_rule_prior[None, :] + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)
    w_rules = pt.exp(log_rules - shift[:, None])

    # Rule plus anchored neighbourhood, two channels; the teacher's weights apply in each channel.
    log_h_prior = -rho * H_RADII - pm.math.logsumexp(-rho * H_RADII)
    log_w_extra = pt.log((cls_extra_mask * tw[cls_extra - 1]).sum(axis=1))  # (cls,)
    x_ll = pt.logaddexp(
        pt.log1p(-eps) + pt.log(omega) + ex_log_tw[cls_set] - log_w_extra[:, None], log_noise
    )  # (cls, 4)
    k_rule = rule_k[cls_set, cls_rule]
    log_cls = (
        pt.log(lambda_0) + pt.log(alpha) + log_rule_prior[cls_rule] + log_h_prior[cls_h]
        - pt.log(cls_m)
        + beta * (
            side_in_ll[cls_set, cls_rule]
            + (cls_ex_in * x_ll).sum(axis=1)
            + (set_n[cls_set] - k_rule - cls_e) * log_noise
        )
    )
    w_cls = pt.exp(log_cls - shift[cls_set])
    w_aug = _scatter_sum(n_sets * N_RULES, cls_set * N_RULES + cls_rule, w_cls).reshape((n_sets, N_RULES))

    # Intervals: teacher shows x in [a, b] with weight D(x)^-tau / sum over [a, b].
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(tw)])
    log_w_int = pt.log(cum_w[INT_B] - cum_w[INT_A - 1])
    pair_lik = (
        pair_k * (pt.log1p(-eps) - log_w_int[pair_int])
        - tau * (pair_log_d - pair_k * LOG_D.mean())
        + (set_n[pair_set] - pair_k) * log_noise
    )
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z
    w_pairs = pt.exp(log_size_prior[pt.as_tensor(INT_SIZE_IDX)[pair_int]] + beta * pair_lik - shift[pair_set])
    size_prior = pt.exp(log_size_prior)
    empty_scale = pt.exp(beta * set_n * log_noise - shift)
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)

    z = w_rules.sum(axis=1) + w_aug.sum(axis=1) + _scatter_sum(n_sets, pair_set, w_pairs) + w_empty
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
        + (w_aug[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, chit_cell, w_cls[chit_cls])
        + _scatter_sum(n_cells, hit_cell, w_pairs[hit_pair])
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

    ex_num = np.ones((len(unique), MAX_EX), dtype="int64")
    ex_mask = np.zeros((len(unique), MAX_EX))
    for i, st in enumerate(unique):
        ex_num[i, : len(st)] = st
        ex_mask[i, : len(st)] = 1.0
    ex_in_rule = RULE_MEMBERSHIP[:, ex_num - 1].transpose(1, 2, 0) * ex_mask[:, :, None]

    pair_set, pair_int, pair_k, pair_log_d, empty_size_count = [], [], [], [], []
    set_pairs, set_empty = [], []
    offset = 0
    for i, st in enumerate(unique):
        ex = np.array(st)
        inside = (INT_A[:, None] <= ex[None, :]) & (INT_B[:, None] >= ex[None, :])
        k = inside.sum(axis=1)
        listed = np.nonzero(k > 0)[0]
        empty = np.nonzero(k == 0)[0]
        pair_set += [i] * len(listed)
        pair_int += list(listed)
        pair_k += list(k[listed].astype(float))
        pair_log_d += list((inside[listed] * LOG_D[ex - 1][None, :]).sum(axis=1))
        empty_size_count.append(np.bincount(INT_SIZE_IDX[empty], minlength=100).astype("float64"))
        set_pairs.append((offset, listed))
        set_empty.append(empty)
        offset += len(listed)

    hit_pair, hit_cell, cell_empty_size_count = [], [], []
    for c, (i, t) in enumerate(cells):
        start, listed = set_pairs[i]
        has_t = np.nonzero((INT_A[listed] <= t) & (INT_B[listed] >= t))[0]
        hit_pair += list(start + has_t)
        hit_cell += [c] * len(has_t)
        empty = set_empty[i]
        holds = empty[(INT_A[empty] <= t) & (INT_B[empty] >= t)]
        cell_empty_size_count.append(np.bincount(INT_SIZE_IDX[holds], minlength=100).astype("float64"))

    cls_rows, cls_extra, cls_extra_mask, cls_ex_in, cls_sets = [], [], [], [], []
    set_cls = []
    for i, st in enumerate(unique):
        mine = []
        for r in range(N_RULES):
            in_rule = RULE_MEMBERSHIP[r] > 0
            centres = sorted({x for x in st if not in_rule[x - 1]})
            for x in centres:
                for h in range(H_MAX + 1):
                    extra = np.arange(max(1, x - h), min(100, x + h) + 1)
                    extra = extra[~in_rule[extra - 1]]
                    row = np.ones(MAX_EXTRA, dtype="int64")
                    row[: len(extra)] = extra
                    msk = np.zeros(MAX_EXTRA)
                    msk[: len(extra)] = 1.0
                    exin = np.zeros(MAX_EX)
                    for j, y in enumerate(st):
                        if y in set(extra.tolist()):
                            exin[j] = 1.0
                    mine.append(len(cls_rows))
                    cls_rows.append((i, r, h, exin.sum(), len(centres)))
                    cls_extra.append(row)
                    cls_extra_mask.append(msk)
                    cls_ex_in.append(exin)
                    cls_sets.append(set(extra.tolist()))
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
        "ex_mask": ex_mask,
        "ex_in_rule": ex_in_rule,
        "cls_set": cls[:, 0].astype("int64"),
        "cls_rule": cls[:, 1].astype("int64"),
        "cls_h": cls[:, 2].astype("int64"),
        "cls_e": cls[:, 3],
        "cls_m": cls[:, 4],
        "cls_extra": np.stack(cls_extra),
        "cls_extra_mask": np.stack(cls_extra_mask),
        "cls_ex_in": np.stack(cls_ex_in),
        "chit_cell": np.array(chit_cell, dtype="int64"),
        "chit_cls": np.array(chit_cls, dtype="int64"),
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "pair_set": np.array(pair_set, dtype="int64"),
        "pair_int": np.array(pair_int, dtype="int64"),
        "pair_k": np.array(pair_k, dtype="float64"),
        "pair_log_d": np.array(pair_log_d, dtype="float64"),
        "empty_size_count": np.stack(empty_size_count),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_empty_size_count": np.stack(cell_empty_size_count),
        "hit_pair": np.array(hit_pair, dtype="int64"),
        "hit_cell": np.array(hit_cell, dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
