"""Rule plus a short run of exceptions, in exception-tolerant tempered concept learning.

Refinement of `rule_plus_exception_concepts`: the same 45 rules (final-digit rules
boosted by kappa), Erlang-width intervals with compressed-number-line sampling,
tempered size principle (beta), exception rate eps, and a share alpha of each
rule's prior given to rule-with-exception concepts. The one change: the exception
added to a rule R is a short run of consecutive numbers [a, b] (width 1..MAX_RUN,
at least one number outside R), not a single number. Run width s has prior
proportional to exp(-(s - 1) / tau), split equally over the runs of that width;
tau -> 0 gives back the single-number exception. Because a run around an
off-pattern example adds few numbers to a broad rule (a run around an odd number
adds only odd numbers to "even"), people endorse the example's neighbours too.
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
MAX_RUN = 10  # longest run of exceptions a rule can carry
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)  # intervals of each width


# Exception runs [a, b] of width 1..MAX_RUN, and how many numbers each adds to each rule.
RUN_A = np.array([a for s in range(1, MAX_RUN + 1) for a in range(1, 102 - s)])
RUN_W = np.array([s for s in range(1, MAX_RUN + 1) for a in range(1, 102 - s)])
RUN_MASK = ((NUMBERS[None, :] >= RUN_A[:, None]) & (NUMBERS[None, :] < (RUN_A + RUN_W)[:, None])).astype("float64")
OFF_RULE = 1.0 - RULE_MEMBERSHIP  # (rules, numbers)
RUN_ADD = (RUN_MASK @ OFF_RULE.T).astype("int64")  # (runs, rules): numbers added, m
# Concepts R + run are grouped by (width s, added m); count of runs per (rule, s, m).
N_SM = MAX_RUN * MAX_RUN
RUN_KEY = (RUN_W - 1)[:, None] * MAX_RUN + (RUN_ADD - 1)  # (runs, rules), valid where m >= 1
RUN_VALID = RUN_ADD >= 1
RUN_COUNT = np.zeros((N_RULES, N_SM))
for _r in range(N_RULES):
    np.add.at(RUN_COUNT[_r], RUN_KEY[RUN_VALID[:, _r], _r], 1.0)
RUNS_PER_WIDTH = np.maximum(RUN_COUNT.reshape(N_RULES, MAX_RUN, MAX_RUN).sum(axis=2), 1.0)  # (rules, s)
KEY_S = np.repeat(np.arange(1, MAX_RUN + 1), MAX_RUN).astype("float64")  # (s, m) flattened
KEY_M = np.tile(np.arange(1, MAX_RUN + 1), MAX_RUN).astype("float64")


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    # Intervals holding >= 1 example, one entry per (set, interval).
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_int = pm.Data("pair_int", np.zeros(1, dtype="int64"))
    pair_k = pm.Data("pair_k", np.zeros(1))  # examples inside the interval
    pair_log_x = pm.Data("pair_log_x", np.zeros(1))  # sum of log x over those examples
    # Intervals holding no example all share one likelihood: lumped per set.
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 100)))  # (sets, widths)
    # Cells (set, target): which listed intervals contain the target.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_empty_size_count = pm.Data("cell_empty_size_count", np.zeros((1, 100)))
    hit_pair = pm.Data("hit_pair", np.zeros(1, dtype="int64"))
    hit_cell = pm.Data("hit_cell", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    # Rule + run concepts. Base: runs holding no off-rule example, counted per (rule, s, m).
    # Runs that hold e >= 1 off-rule examples are listed as sparse groups (set or cell, rule, s-m key, e, count).
    cell_run_count = pm.Data("cell_run_count", np.zeros((1, N_RULES, N_SM)))  # runs holding the target (target off rule)
    grp_set = pm.Data("grp_set", np.zeros(1, dtype="int64"))
    grp_flat = pm.Data("grp_flat", np.zeros(1, dtype="int64"))  # flat (rule, key) index
    grp_e = pm.Data("grp_e", np.zeros(1))
    grp_cnt = pm.Data("grp_cnt", np.zeros(1))
    cgrp_cell = pm.Data("cgrp_cell", np.zeros(1, dtype="int64"))
    cgrp_set = pm.Data("cgrp_set", np.zeros(1, dtype="int64"))
    cgrp_flat = pm.Data("cgrp_flat", np.zeros(1, dtype="int64"))
    cgrp_e = pm.Data("cgrp_e", np.zeros(1))
    cgrp_cnt = pm.Data("cgrp_cnt", np.zeros(1))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)  # typical interval width
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)  # log prior boost of final-digit rules
    alpha = pm.Beta("alpha", 1, 4)  # share of a rule's prior given to its rule-plus-run variants
    tau = pm.LogNormal("tau", mu=np.log(1.0), sigma=1.0)  # how fast run plausibility falls with width

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)  # (rules,)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rule_prior = kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    log_rules = pt.log(lambda_0) + pt.log1p(-alpha) + log_rule_prior[None, :] + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)  # per-set stabiliser
    w_rules = pt.exp(log_rules - shift[:, None])

    # Rule R plus a run of width s adding m numbers: size |R| + m.
    log_q = -(np.arange(MAX_RUN) / tau)
    log_q = log_q - pm.math.logsumexp(log_q)  # (s,)
    log_run_prior = log_q[None, :] - np.log(RUNS_PER_WIDTH)  # (rules, s): one run of width s
    run_in = pt.log((1 - eps) / (RULE_SIZE[:, None] + KEY_M[None, :]) + eps / 100)  # (rules, keys)
    run_gain = beta * (run_in - log_noise[None])  # extra log weight per off-rule example inside the run
    log_aug0 = (
        pt.log(lambda_0)
        + pt.log(alpha)
        + log_rule_prior[:, None]
        + log_run_prior[:, (KEY_S - 1).astype("int64")]
    )  # (rules, keys)
    log_a = (
        log_aug0[None]
        + beta * (rule_k[:, :, None] * run_in[None] + (set_n[:, None, None] - rule_k[:, :, None]) * log_noise)
        - shift[:, None, None]
    )  # (sets, rules, keys): runs holding no off-rule example
    w_a = pt.exp(log_a)
    run_gain_flat = run_gain.flatten()
    log_a_flat = log_a.reshape((n_sets, N_RULES * N_SM))

    def _group_extra(gset, gflat, ge, gcnt):
        base = log_a_flat[gset, gflat]
        return gcnt * (pt.exp(base + ge * run_gain_flat[gflat]) - pt.exp(base))

    w_aug_base = (w_a * RUN_COUNT[None]).sum(axis=2)  # (sets, rules)
    extra = _group_extra(grp_set, grp_flat, grp_e, grp_cnt)
    w_aug = w_aug_base + _scatter_sum(n_sets * N_RULES, grp_set * N_RULES + grp_flat // N_SM, extra).reshape(
        (n_sets, N_RULES)
    )

    # Interval members are drawn with weight x^-gamma: P(x | [a, b]) = x^-gamma / W(a, b).
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    log_w = pt.log(cum_w[INT_B] - cum_w[INT_A - 1])  # (intervals,)
    pair_lik = (
        pair_k * (pt.log1p(-eps) - log_w[pair_int])
        - gamma * pair_log_x
        + (set_n[pair_set] - pair_k) * log_noise
    )
    # Erlang width prior: log P(interval of width s) = log s - s / sigma - log Z.
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z  # (widths,)
    w_pairs = pt.exp(log_size_prior[pt.as_tensor(INT_SIZE_IDX)[pair_int]] + beta * pair_lik - shift[pair_set])
    size_prior = pt.exp(log_size_prior)
    empty_scale = pt.exp(beta * set_n * log_noise - shift)  # (sets,): shared likelihood of empty intervals
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)

    z = w_rules.sum(axis=1) + w_aug.sum(axis=1) + _scatter_sum(n_sets, pair_set, w_pairs) + w_empty  # (sets,)
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
        + (w_aug[cell_set] * cell_rule_member).sum(axis=1)
        + (w_a[cell_set] * cell_run_count).sum(axis=(1, 2))
        + _scatter_sum(n_cells, cgrp_cell, _group_extra(cgrp_set, cgrp_flat, cgrp_e, cgrp_cnt))
        + _scatter_sum(n_cells, hit_cell, w_pairs[hit_pair])
        + empty_scale[cell_set] * (cell_empty_size_count * size_prior[None, :]).sum(axis=1)
    )
    in_concept = numer / z[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _groups(owner, run_e, run_sel):
    """Group runs with e >= 1 off-rule examples by (rule, key, e): returns owner, flat, e, count lists."""
    out = {}
    runs, rules = np.nonzero(run_sel & RUN_VALID & (run_e > 0))
    for j, r in zip(runs, rules):
        key = (r * N_SM + RUN_KEY[j, r], run_e[j, r])
        out[key] = out.get(key, 0) + 1
    return [(owner, f, float(e), float(c)) for (f, e), c in out.items()]


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}

    pair_set, pair_int, pair_k, pair_log_x, empty_size_count = [], [], [], [], []
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
        pair_log_x += list((inside[listed] * np.log(ex)[None, :]).sum(axis=1))
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

    # Off-rule examples (with multiplicity) inside each run, per set.
    run_e = []
    groups = []
    for i, st in enumerate(unique):
        counts = np.bincount(np.array(st) - 1, minlength=100).astype("float64")
        e = (RUN_MASK @ (OFF_RULE * counts[None, :]).T).astype("int64")  # (runs, rules)
        run_e.append(e)
        groups += _groups(i, e, np.ones_like(e, dtype=bool))
    cell_run_count = np.zeros((len(cells), N_RULES, N_SM))
    cgroups = []
    for c, (i, t) in enumerate(cells):
        holds = (RUN_MASK[:, t - 1] > 0)[:, None] & (OFF_RULE[:, t - 1] > 0)[None, :]  # (runs, rules)
        runs, rules = np.nonzero(holds & RUN_VALID)  # counted at their e = 0 weight; groups add the rest
        np.add.at(cell_run_count[c], (rules, RUN_KEY[runs, rules]), 1.0)
        cgroups += [(c, f, e, n) for _, f, e, n in _groups(i, run_e[i], holds)]

    def col(g, k, dtype):
        return np.array([x[k] for x in g], dtype=dtype) if g else np.zeros(0, dtype=dtype)

    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "pair_set": np.array(pair_set, dtype="int64"),
        "pair_int": np.array(pair_int, dtype="int64"),
        "pair_k": np.array(pair_k, dtype="float64"),
        "pair_log_x": np.array(pair_log_x, dtype="float64"),
        "empty_size_count": np.stack(empty_size_count),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_empty_size_count": np.stack(cell_empty_size_count),
        "hit_pair": np.array(hit_pair, dtype="int64"),
        "hit_cell": np.array(hit_cell, dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "cell_run_count": cell_run_count,
        "grp_set": col(groups, 0, "int64"),
        "grp_flat": col(groups, 1, "int64"),
        "grp_e": col(groups, 2, "float64"),
        "grp_cnt": col(groups, 3, "float64"),
        "cgrp_cell": col(cgroups, 0, "int64"),
        "cgrp_set": np.array([cells[x[0]][0] for x in cgroups], dtype="int64"),
        "cgrp_flat": col(cgroups, 1, "int64"),
        "cgrp_e": col(cgroups, 2, "float64"),
        "cgrp_cnt": col(cgroups, 3, "float64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
