"""Tempered concept learning with lapses to guessing (refinement of
tempered_size_principle_concept_learning).

People do Bayesian concept learning over 42 rules and all intervals of 1..100,
applying the size principle only partially (examples' log-likelihood scaled by
an inferred evidence strength gamma). The single change: on an inferred
fraction of trials people do not consult the inferred concept and instead
guess, answering "yes" with an inferred guessing bias. So
P(yes) = (1 - lapse) * P(target in concept) + lapse * guess_bias.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_EX = 4
N_GROUPS = MAX_EX * (MAX_EX + 1) // 2 + 1  # runs of sorted examples, plus "none"


def rule_sets():
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    return rules


RULES = rule_sets()
N_RULES = len(RULES)
N_INTERVALS = 5050
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(RULE_MEMBERSHIP * (1 - EPSILON) / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
SIZES = np.arange(1, 101)
LOG_IN = np.log((1 - EPSILON) / SIZES + EPSILON / 100)  # log P(x | interval of this size), x inside
LOG_OUT = np.log(EPSILON / 100)

# Under a tempered likelihood an interval's likelihood depends only on its size and
# which examples it contains (always a run of the sorted examples). So, per
# example set, intervals are pooled exactly into (run, size) cells: a cell's
# prior is its interval count, and its in-concept probability for a target the
# share of its intervals containing that target. prepare_observed builds the
# non-empty cells of each set (padded to a common length with empty cells) and
# the in-concept shares only for the (set, target) pairs actually asked about.


def _set_cells(s):
    """Non-empty (run, size) cells of set s: count, size index, run, (cells, 100) shares."""
    order = sorted(range(len(s)), key=lambda k: s[k])
    vals = [s[k] for k in order]
    runs = [(i, j) for i in range(len(vals)) for j in range(i, len(vals))]
    run_id = {r: g + 1 for g, r in enumerate(runs)}
    counts = np.zeros((N_GROUPS, 100))
    in_counts = np.zeros((N_GROUPS, 100, 100))
    for m in range(1, 101):
        for n in range(m, 101):
            inside = [i for i, v in enumerate(vals) if m <= v <= n]
            g = run_id[(inside[0], inside[-1])] if inside else 0
            counts[g, n - m] += 1
            in_counts[g, n - m, m - 1:n] += 1
    member = np.zeros((N_GROUPS, MAX_EX))  # run -> presented positions it contains
    for (i, j), g in run_id.items():
        for k in order[i:j + 1]:
            member[g, k] = 1.0
    g_idx, size_idx = np.nonzero(counts)
    c = counts[g_idx, size_idx]
    return c, size_idx, member[g_idx], in_counts[g_idx, size_idx] / c[:, None]


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    cells = [_set_cells(s) for s in unique]
    K = max(len(c[0]) for c in cells)
    rule_ll = np.zeros((S, MAX_EX, N_RULES))
    lag = np.zeros((S, MAX_EX))
    present = np.zeros((S, MAX_EX))
    log_count = np.full((S, K), -1e4)  # padding cells: prior ~0
    cell_log_in = np.zeros((S, K))
    cell_member = np.zeros((S, K, MAX_EX))
    shares = np.zeros((S, K, 100))
    for i, s in enumerate(unique):
        for k, x in enumerate(s):
            rule_ll[i, k] = RULE_LOG_LIK[:, x - 1]
            lag[i, k] = len(s) - 1 - k
            present[i, k] = 1.0
        c, size_idx, mem, sh = cells[i]
        n = len(c)
        log_count[i, :n] = np.log(c)
        cell_log_in[i, :n] = LOG_IN[size_idx]
        cell_member[i, :n] = mem
        shares[i, :n] = sh
    pairs = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    queries = list(dict.fromkeys(pairs))
    q_index = {q: j for j, q in enumerate(queries)}
    q_set = np.array([q[0] for q in queries], dtype="int64")
    q_target = np.array([q[1] for q in queries], dtype="int64")
    return {
        "rule_ex_log_lik": rule_ll,
        "example_lag": lag,
        "example_present": present,
        "cell_log_count": log_count,
        "cell_log_in": cell_log_in,
        "cell_member": cell_member,
        "query_set": q_set,
        "query_rule_member": RULE_MEMBERSHIP[:, q_target].T,
        "query_cell_in_concept": shares[q_set, :, q_target],
        "row_query": np.array([q_index[p] for p in pairs], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_ex_log_lik = pm.Data("rule_ex_log_lik", np.zeros((1, MAX_EX, N_RULES)))  # log P(x_k | rule)
    example_lag = pm.Data("example_lag", np.zeros((1, MAX_EX)))  # unused by this hypothesis
    example_present = pm.Data("example_present", np.zeros((1, MAX_EX)))
    cell_log_count = pm.Data("cell_log_count", np.zeros((1, 1)))  # (sets, cells) log #intervals
    cell_log_in = pm.Data("cell_log_in", np.zeros((1, 1)))  # log P(x | interval of the cell's size)
    cell_member = pm.Data("cell_member", np.zeros((1, 1, MAX_EX)))  # cell contains example k
    query_set = pm.Data("query_set", np.zeros(1, dtype="int64"))
    query_rule_member = pm.Data("query_rule_member", np.zeros((1, N_RULES)))
    query_cell_in_concept = pm.Data("query_cell_in_concept", np.zeros((1, 1)))
    row_query = pm.Data("row_query", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    # Evidence strength of the examples (1 = full size principle, the incumbent).
    gamma = pm.LogNormal("gamma", mu=0.0, sigma=0.5)

    w = gamma * example_present  # (sets, positions): same weight for every example
    rule_ll = (w[:, :, None] * rule_ex_log_lik).sum(axis=1)  # (sets, rules)
    w_in = (cell_member * w[:, None, :]).sum(axis=2)  # (sets, cells)
    w_out = w.sum(axis=1)[:, None] - w_in
    cell_ll = w_in * cell_log_in + w_out * LOG_OUT

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_ll
    log_cells = pt.log1p(-p_mix) - np.log(N_INTERVALS) + cell_log_count + cell_ll
    shift = pt.maximum(log_rules.max(axis=1), log_cells.max(axis=1))[:, None]
    u_rules = pt.exp(log_rules - shift)
    u_cells = pt.exp(log_cells - shift)
    z = u_rules.sum(axis=1) + u_cells.sum(axis=1)  # (sets,)
    in_concept = (
        (u_rules[query_set] * query_rule_member).sum(axis=1)
        + (u_cells[query_set] * query_cell_in_concept).sum(axis=1)
    ) / z[query_set]

    # Lapse to guessing: on a fraction of trials the concept is not consulted.
    lapse = pm.Beta("lapse", 2.0, 4.0)
    guess_bias = pm.Beta("guess_bias", 2.0, 2.0)
    p_concept = (1 - lapse) * in_concept[row_query] + lapse * guess_bias
    p_yes = pm.Deterministic("p_yes", pt.clip(p_concept, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
