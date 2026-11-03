"""Span-filling example evidence: refinement of position_decaying_example_evidence.

People read the examples one after another and build a running guess about the
program; as in the incumbent, the log-evidence the k-th example carries is
scaled by exp(rho * (k - 1)), rho inferred. The one change: how much an example
revises the guess also depends on whether it extends what was seen before. An
example that falls inside the span of the examples shown before it (e.g. 15
after 31, 3, 1) adds no new extent, and its evidence is scaled by exp(-gamma),
gamma inferred (gamma > 0: such fillers are treated as redundant and narrow
generalization less; gamma < 0: a filler confirms the pattern and narrows it
more). Otherwise this is plain Bayesian concept learning: 45 arithmetic and
digit rules, the 18 digit-sum concepts (inferred family share) and every
interval of 1..100 under an equal prior, examples read as independent draws
(with replacement) with 10% strays, and a guessing lapse.

Speed: under the equal interval prior, an interval's weighted evidence depends
only on which examples it covers (a contiguous run of the sorted examples) and
its width, so prepare_observed counts intervals per (coverage pattern, width)
for each set, and per (set, target) cell those that contain the target.
"""

import collections

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
STRAY = (1 - ALPHA) / 100
NUMBERS = np.arange(1, 101)
MAX_EX = 4
MAX_PAT = MAX_EX * (MAX_EX + 1) // 2 + 1  # contiguous runs of sorted examples, plus none


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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
DIGIT_SUM = np.array([sum(int(c) for c in str(n)) for n in NUMBERS])
SUM_CONCEPTS = [list(NUMBERS[DIGIT_SUM == s]) for s in range(1, 19)]
N_SUMS = len(SUM_CONCEPTS)
FINITE_MEM = np.array([np.isin(NUMBERS, c) for c in RULES + SUM_CONCEPTS], dtype="float64")
FINITE_SIZE = FINITE_MEM.sum(axis=1)
# log gain of an example a concept covers over a stray draw, by concept size
FINITE_GAIN = np.log(ALPHA / FINITE_SIZE + STRAY) - np.log(STRAY)
WIDTH_GAIN = np.log(ALPHA / NUMBERS + STRAY) - np.log(STRAY)  # interval widths 1..100
N_INTERVALS = 100 * 101 // 2

with pm.Model() as model:
    finite_cover = pm.Data("finite_cover", np.zeros((1, MAX_EX, N_RULES + N_SUMS)))  # (sets, pos, concepts)
    example_mask = pm.Data("example_mask", np.zeros((1, MAX_EX)))  # (sets, pos)
    inside_span = pm.Data("inside_span", np.zeros((1, MAX_EX)))  # (sets, pos) 1 if within earlier examples' span
    pattern_cover = pm.Data("pattern_cover", np.zeros((1, MAX_PAT, MAX_EX)))  # (sets, pattern, pos)
    class_set = pm.Data("class_set", np.zeros(1, dtype="int64"))  # (classes,) interval class: set,
    class_pattern = pm.Data("class_pattern", np.zeros(1, dtype="int64"))  # coverage pattern,
    class_gain = pm.Data("class_gain", np.zeros(1))  # per-example gain of its width,
    class_count = pm.Data("class_count", np.zeros(1))  # and number of intervals in it
    entry_class = pm.Data("entry_class", np.zeros(1, dtype="int64"))  # (entries,) class of a cell entry
    entry_cell = pm.Data("entry_cell", np.zeros(1, dtype="int64"))  # (entries,) cell of the entry
    entry_count = pm.Data("entry_count", np.zeros(1))  # (entries,) intervals of the class holding the target
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # (cells,) 0-based
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)
    rho = pm.Normal("rho", 0.0, 0.5)  # per-position log change in an example's evidence weight
    gamma = pm.Normal("gamma", 0.0, 1.0)  # log discount of an example that adds no new extent

    weights = (
        example_mask
        * pt.exp(rho * np.arange(MAX_EX, dtype="float64"))[None, :]
        * pt.exp(-gamma * inside_span)
    )  # (sets, pos)

    finite_ll = (finite_cover * weights[:, :, None]).sum(axis=1) * FINITE_GAIN[None, :]
    rule_ll, sum_ll = finite_ll[:, :N_RULES], finite_ll[:, N_RULES:]
    rule_ev = pm.math.logsumexp(rule_ll, axis=1)
    sum_ev = pm.math.logsumexp(sum_ll, axis=1)
    rule_in = pt.dot(pt.exp(rule_ll - rule_ev[:, None]), FINITE_MEM[:N_RULES])  # (sets, 100)
    sum_in = pt.dot(pt.exp(sum_ll - sum_ev[:, None]), FINITE_MEM[N_RULES:])

    pattern_weight = (pattern_cover * weights[:, None, :]).sum(axis=2)  # (sets, pattern)
    # shift by an upper bound on each set's largest interval log-likelihood (gain is largest at width 1)
    shift = pattern_weight.max(axis=1) * WIDTH_GAIN[0]
    class_ll = pattern_weight[class_set, class_pattern] * class_gain - shift[class_set]
    n_sets = pattern_weight.shape[0]
    mass = pt.inc_subtensor(pt.zeros((n_sets,))[class_set], class_count * pt.exp(class_ll))
    int_ev = pt.log(mass) + shift
    class_post = pt.exp(class_ll - pt.log(mass)[class_set])  # posterior of one interval in the class
    n_cells = cell_set.shape[0]
    interval_in_cell = pt.inc_subtensor(pt.zeros((n_cells,))[entry_cell], entry_count * class_post[entry_class])

    log_rules_only = rule_ev - np.log(N_RULES)
    log_sums_only = sum_ev - np.log(N_SUMS) + log_sum_share
    p_sum = pm.math.sigmoid(log_sums_only - log_rules_only)[cell_set]
    finite_in_cell = (1 - p_sum) * rule_in[cell_set, cell_target] + p_sum * sum_in[cell_set, cell_target]
    log_finite = (
        pt.log(lambda_0) + pt.logaddexp(log_rules_only, log_sums_only) - pt.logaddexp(0.0, log_sum_share)
    )
    log_intervals = pt.log1p(-lambda_0) + int_ev - np.log(N_INTERVALS)
    p_rule = pm.math.sigmoid(log_finite - log_intervals)[cell_set]
    in_concept = p_rule * finite_in_cell + (1 - p_rule) * interval_in_cell

    p_concept = pt.clip(in_concept[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def interval_patterns(examples):
    """Coverage-pattern index and width index for every interval [lo, hi], plus each pattern's positions."""
    lo = np.repeat(NUMBERS, 101 - NUMBERS)
    hi = np.concatenate([np.arange(m, 101) for m in NUMBERS])
    ex = np.array(examples)
    covers = (lo[:, None] <= ex[None, :]) & (hi[:, None] >= ex[None, :])  # (intervals, pos)
    patterns, index = np.unique(covers, axis=0, return_inverse=True)
    return lo, hi, index.reshape(-1), patterns.astype("float64"), hi - lo


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    keys = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    n_sets, n_cells = len(unique), len(cells)
    finite_cover = np.zeros((n_sets, MAX_EX, N_RULES + N_SUMS))
    mask = np.zeros((n_sets, MAX_EX))
    inside = np.zeros((n_sets, MAX_EX))
    pat_cover = np.zeros((n_sets, MAX_PAT, MAX_EX))
    classes = {}  # (set, pattern, width) -> class index
    class_count = []
    entries = collections.Counter()  # (cell, class) -> intervals holding the target
    tables = []
    for i, s in enumerate(unique):
        for k, x in enumerate(s):
            mask[i, k] = 1.0
            inside[i, k] = float(k > 0 and min(s[:k]) <= x <= max(s[:k]))
            finite_cover[i, k] = FINITE_MEM[:, x - 1]
        lo, hi, pat, patterns, width = interval_patterns(s)
        pat_cover[i, : len(patterns), : len(s)] = patterns
        ids = np.zeros(len(lo), dtype="int64")
        for j, key in enumerate(zip(pat.tolist(), width.tolist())):
            if (i,) + key not in classes:
                classes[(i,) + key] = len(classes)
                class_count.append(0.0)
            ids[j] = classes[(i,) + key]
            class_count[ids[j]] += 1.0
        tables.append((lo, hi, ids))
    for c, (si, t) in enumerate(cells):
        lo, hi, ids = tables[si]
        for cls in ids[(lo <= t) & (hi >= t)].tolist():
            entries[(c, cls)] += 1.0
    keys_cls = list(classes)
    entry_keys = list(entries)
    return {
        "finite_cover": finite_cover,
        "example_mask": mask,
        "inside_span": inside,
        "pattern_cover": pat_cover,
        "class_set": np.array([k[0] for k in keys_cls], dtype="int64"),
        "class_pattern": np.array([k[1] for k in keys_cls], dtype="int64"),
        "class_gain": np.array([WIDTH_GAIN[k[2]] for k in keys_cls]),
        "class_count": np.array(class_count),
        "entry_class": np.array([e[1] for e in entry_keys], dtype="int64"),
        "entry_cell": np.array([e[0] for e in entry_keys], dtype="int64"),
        "entry_count": np.array([entries[e] for e in entry_keys]),
        "cell_set": np.array([c[0] for c in cells], dtype="int64"),
        "cell_target": np.array([c[1] - 1 for c in cells], dtype="int64"),
        "row_cell": np.array([{c: j for j, c in enumerate(cells)}[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
