"""Anchored side-channel exceptions: a neighbourhood or, as the next-larger region, the outlier's decade.

Refinement of `iter0_candidate3`: the same 45 rules (final-digit rules boosted
by kappa), Erlang-width intervals with compressed-number-line sampling,
tempered size principle (beta), exception rate eps, and a share alpha of each
rule's prior for exception variants anchored on a shown example x that the
rule misses (split equally over those m examples), the exception region being
a fuzzy neighbourhood [x - h, x + h] (h = 0..3) or the whole decade x falls in,
ranked as the next-larger region (rank 4); P(rank) is proportional to
exp(-rho rank), one fall-off for every region size.
The one change, grafted from `anchored_neighbourhood_side_channel`: such a
concept generates its outputs through two channels -- a rule member with
probability 1 - omega, a member of the exception region (uniformly over its
off-rule numbers) with probability omega -- instead of sampling uniformly from
rule and region pooled together. A whole-decade exception then dilutes only
the outlier's own likelihood, not the rule examples', so people credit the
outlier's whole decade more readily and endorse its same-decade neighbours
(even four to nine away) over off-rule neighbours of the majority examples,
but not its neighbours just across a decade boundary.
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
# Intervals [a, b] of 1..100 as a dense (a, b) grid; VALID marks b >= a.
GRID_A, GRID_B = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
VALID = (GRID_B >= GRID_A).astype("float64")
SAFE_B = np.maximum(GRID_A, GRID_B)  # a dummy [a, a] where b < a
WIDTH_IDX = SAFE_B - GRID_A  # width - 1
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)  # intervals of each width
K_MAX = 4  # most examples in a set
K_RANGE = np.arange(K_MAX + 1, dtype="float64")
H_MAX = 3  # largest exception-neighbourhood radius
N_REGIONS = H_MAX + 2  # radii 0..3, then the whole decade
REGION_RANK = np.arange(N_REGIONS, dtype="float64")
DECADE = np.minimum(NUMBERS // 10, 9)  # decades 1-9, 10-19, ..., 90-100


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    # Intervals holding examples, grouped by which examples they hold: intervals holding exactly the
    # sorted distinct examples i..j form a rectangle a0 <= a <= a1, b0 <= b <= b1 of the (a, b) grid.
    grp_set = pm.Data("grp_set", np.zeros(1, dtype="int64"))
    grp_k = pm.Data("grp_k", np.ones(1, dtype="int64"))  # examples held (with repeats)
    grp_log_x = pm.Data("grp_log_x", np.zeros(1))  # sum of log x over them
    grp_rect = pm.Data("grp_rect", np.ones((1, 4), dtype="int64"))  # a0, a1, b0, b1
    # Intervals holding no example all share one likelihood: lumped per set, by width.
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 100)))
    # Cells (set, target); per (cell, group) the sub-rectangle of the group's intervals holding the target.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_empty_size_count = pm.Data("cell_empty_size_count", np.zeros((1, 100)))
    cg_cell = pm.Data("cg_cell", np.zeros(1, dtype="int64"))
    cg_grp = pm.Data("cg_grp", np.zeros(1, dtype="int64"))
    cg_rect = pm.Data("cg_rect", np.ones((1, 4), dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    # Rule-plus-exception concepts, dense over (set, rule, anchor slot, region): the anchor slots hold the
    # set's distinct examples; a concept exists only where that example lies outside the rule.
    exc_valid = pm.Data("exc_valid", np.zeros((1, N_RULES, K_MAX, N_REGIONS)))
    exc_extra_n = pm.Data("exc_extra_n", np.ones((1, N_RULES, K_MAX, N_REGIONS)))  # off-rule numbers in region
    exc_e = pm.Data("exc_e", np.zeros((1, N_RULES, K_MAX, N_REGIONS)))  # examples in region, outside rule
    exc_log_m = pm.Data("exc_log_m", np.zeros((1, N_RULES)))  # log of the number of possible anchors
    # Sparse (cell, flat concept index): concepts whose exception region holds a target outside the rule.
    chit_cell = pm.Data("chit_cell", np.zeros(1, dtype="int64"))
    chit_exc = pm.Data("chit_exc", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)  # typical interval width
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)  # log prior boost of final-digit rules
    alpha = pm.Beta("alpha", 1, 4)  # share of a rule's prior given to its exception variants
    rho = pm.LogNormal("rho", mu=0.0, sigma=1.0)  # how fast prior falls with the exception region's rank
    omega = pm.Beta("omega", 2, 6)  # rate at which the program emits its exception rather than its rule

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

    # Rule R plus region N (radius 0..3 or the decade) around a shown example x not in R, as two channels:
    # rule members w.p. 1 - omega, members of N \ R uniformly w.p. omega.
    log_region_prior = -rho * REGION_RANK - pm.math.logsumexp(-rho * REGION_RANK)
    side_rule_in = pt.log((1 - eps) * (1 - omega) / RULE_SIZE + eps / 100)  # (rules,)
    # Per (set, rule): everything but the region's own term.
    log_exc_base = (
        pt.log(lambda_0) + pt.log(alpha) + log_rule_prior[None, :] - exc_log_m
        + beta * (rule_k * side_rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise)
        - shift[:, None]
    )
    region_in = pt.log((1 - eps) * omega / exc_extra_n + eps / 100) - log_noise
    log_exc = log_exc_base[:, :, None, None] + log_region_prior[None, None, None, :] + beta * exc_e * region_in
    w_exc = pt.exp(log_exc) * exc_valid  # (sets, rules, anchors, regions)
    w_aug = w_exc.sum(axis=(2, 3))  # (sets, rules)

    # Interval members are drawn with weight x^-gamma: P(x | [a, b]) = x^-gamma / W(a, b).
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    log_w = pt.log(cum_w[SAFE_B] - cum_w[GRID_A - 1])  # (100, 100)
    # Erlang width prior: log P(interval of width s) = log s - s / sigma - log Z.
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z  # (widths,)
    # Per k examples held: prior x W(a, b)^(-beta k) on the grid, and its 2-D prefix sums.
    e_k = pt.exp(log_size_prior[WIDTH_IDX][None] - beta * K_RANGE[:, None, None] * log_w[None]) * VALID[None]
    pref = pt.zeros((K_MAX + 1, 101, 101))
    pref = pt.set_subtensor(pref[:, 1:, 1:], pt.cumsum(pt.cumsum(e_k, axis=1), axis=2))

    def rect_sum(k, r):
        a0, a1, b0, b1 = r[:, 0] - 1, r[:, 1], r[:, 2] - 1, r[:, 3]
        return pref[k, a1, b1] - pref[k, a0, b1] - pref[k, a1, b0] + pref[k, a0, b0]

    grp_scale = pt.exp(
        beta * (grp_k * pt.log1p(-eps) - gamma * grp_log_x + (set_n[grp_set] - grp_k) * log_noise) - shift[grp_set]
    )
    w_grp = grp_scale * rect_sum(grp_k, grp_rect)
    size_prior = pt.exp(log_size_prior)
    empty_scale = pt.exp(beta * set_n * log_noise - shift)  # (sets,): shared likelihood of empty intervals
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)

    w_rule_all = w_rules + w_aug  # a rule and its exception variants all contain the rule's members
    z = w_rule_all.sum(axis=1) + _scatter_sum(n_sets, grp_set, w_grp) + w_empty  # (sets,)
    numer = (
        (w_rule_all[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, chit_cell, w_exc.flatten()[chit_exc])
        + _scatter_sum(n_cells, cg_cell, grp_scale[cg_grp] * rect_sum(grp_k[cg_grp], cg_rect))
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
    n_sets = len(unique)

    grp_rows, grp_rects, set_groups = [], [], []
    empty_size_count, set_empty = [], []
    for i, st in enumerate(unique):
        assert len(st) <= K_MAX
        u, cnt = np.unique(np.array(st), return_counts=True)
        edge = np.concatenate([[0], u, [101]])
        mine = []
        for lo in range(len(u)):
            for hi in range(lo, len(u)):
                mine.append(len(grp_rows))
                grp_rows.append((i, cnt[lo:hi + 1].sum(), (cnt[lo:hi + 1] * np.log(u[lo:hi + 1])).sum()))
                grp_rects.append((edge[lo] + 1, edge[lo + 1], edge[hi + 1], edge[hi + 2] - 1))
        set_groups.append(mine)
        empty = (VALID > 0) & ~((GRID_A[..., None] <= u) & (GRID_B[..., None] >= u)).any(axis=-1)
        set_empty.append(empty)
        empty_size_count.append(np.bincount(WIDTH_IDX[empty], minlength=100).astype("float64"))

    cg_cell, cg_grp, cg_rects, cell_empty_size_count = [], [], [], []
    for c, (i, t) in enumerate(cells):
        for g in set_groups[i]:
            a0, a1, b0, b1 = grp_rects[g]
            a1, b0 = min(a1, t), max(b0, t)
            if a0 <= a1 and b0 <= b1:
                cg_cell.append(c)
                cg_grp.append(g)
                cg_rects.append((a0, a1, b0, b1))
        holds = set_empty[i] & (GRID_A <= t) & (GRID_B >= t)
        cell_empty_size_count.append(np.bincount(WIDTH_IDX[holds], minlength=100).astype("float64"))

    shape = (n_sets, N_RULES, K_MAX, N_REGIONS)
    exc_valid, exc_extra_n, exc_e = np.zeros(shape), np.ones(shape), np.zeros(shape)
    exc_log_m = np.zeros((n_sets, N_RULES))
    exc_members = {}  # flat concept index -> off-rule numbers its region adds
    for i, st in enumerate(unique):
        ex_count = np.bincount(np.array(st), minlength=101)[1:]
        for r in range(N_RULES):
            in_rule = RULE_MEMBERSHIP[r] > 0
            centres = sorted({x for x in st if not in_rule[x - 1]})
            exc_log_m[i, r] = np.log(max(len(centres), 1))
            for slot, x in enumerate(centres):
                for h in range(N_REGIONS):
                    if h == N_REGIONS - 1:
                        extra = NUMBERS[DECADE == DECADE[x - 1]]
                    else:
                        extra = np.arange(max(1, x - h), min(100, x + h) + 1)
                    extra = extra[~in_rule[extra - 1]]
                    exc_valid[i, r, slot, h] = 1.0
                    exc_extra_n[i, r, slot, h] = len(extra)
                    exc_e[i, r, slot, h] = ex_count[extra - 1].sum()
                    exc_members[np.ravel_multi_index((i, r, slot, h), shape)] = set(extra.tolist())
    by_set = {}
    for j in exc_members:
        by_set.setdefault(np.unravel_index(j, shape)[0], []).append(j)
    chit_cell, chit_exc = [], []
    for c, (i, t) in enumerate(cells):
        for j in by_set.get(i, []):
            if t in exc_members[j]:
                chit_cell.append(c)
                chit_exc.append(j)

    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "grp_set": np.array([g[0] for g in grp_rows], dtype="int64"),
        "grp_k": np.array([g[1] for g in grp_rows], dtype="int64"),
        "grp_log_x": np.array([g[2] for g in grp_rows], dtype="float64"),
        "grp_rect": np.array(grp_rects, dtype="int64"),
        "empty_size_count": np.stack(empty_size_count),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_empty_size_count": np.stack(cell_empty_size_count),
        "cg_cell": np.array(cg_cell, dtype="int64"),
        "cg_grp": np.array(cg_grp, dtype="int64"),
        "cg_rect": np.array(cg_rects, dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "exc_valid": exc_valid,
        "exc_extra_n": exc_extra_n,
        "exc_e": exc_e,
        "exc_log_m": exc_log_m,
        "chit_cell": np.array(chit_cell, dtype="int64"),
        "chit_exc": np.array(chit_exc, dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
