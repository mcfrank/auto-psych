"""Refinement of bigelow_piantadosi_interval: Bayesian concept learning over
42 rules and the 5,050 intervals of 1..100 (rule weights inferred, interval
prior a Beta density over the midpoint), but the size principle is applied
with an inferred strength `alpha` in [0, 1]. Each example contributes
(1 - eps) * [x in h] * |h|^-alpha + eps * 100^-alpha to the likelihood of
concept h (eps = 0.1): alpha = 1 is the incumbent's strong sampling exactly,
alpha = 0 is weak sampling (consistency only, with noise). "Yes" is the
posterior probability that the concept contains the target. The single change
from the incumbent is the inferred size-principle strength.

A concept's likelihood depends only on its size and on how many of the
examples it contains, so for each set the intervals are pooled by (size,
count); a pool is a few runs of intervals with consecutive starts, so its
prior mass is read from cumulative sums of the midpoint prior, avoiding a
dense (sets x 5,092) posterior.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


def concept_sets():
    """The reference's `np_utils.get_concept_sets`: 42 rules, then 5,050 intervals."""
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
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
RULE_SIZE_INDEX = RULE_MEMBERSHIP.sum(axis=1).astype("int64") - 1
INTERVAL_SIZE_INDEX = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64") - 1
LOG_SIZES = np.log(np.arange(1, 101, dtype="float64"))

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])
INTERVAL_START = np.array([c[0] for c in INTERVALS]) - 1  # 0-based
# Grid point of the interval of size index L starting at 0-based m, as a
# (100, 101) table (101 = a dummy, density-free column for starts past 100).
SIZE_START_POINT = np.full((100, 101), 199, dtype="int64")
for _size in range(100):
    for _start in range(100 - _size):
        SIZE_START_POINT[_size, _start] = 2 * _start + _size


def _pooling(keys_per_row):
    """Pool intervals by (size, count) key within each row. For one size the
    intervals are ordered by start, so a pool is a few runs of consecutive
    starts: returns each pool's row and key, and its runs as (pool, size,
    first start, last start + 1)."""
    pool_row, pool_key, runs = [], [], []
    for r, keys in enumerate(keys_per_row):
        for key in dict.fromkeys(map(tuple, keys[keys[:, 0] >= 0])):
            members = (keys[:, 0] == key[0]) & (keys[:, 1] == key[1])
            starts = np.sort(INTERVAL_START[members])
            breaks = np.flatnonzero(np.diff(starts) > 1)
            for lo, hi in zip(np.r_[0, breaks + 1], np.r_[breaks, len(starts) - 1]):
                runs.append((len(pool_row), key[0], starts[lo], starts[hi] + 1))
            pool_row.append(r)
            pool_key.append(key)
    runs = np.array(runs, dtype="int64")
    return (
        np.array(pool_row, dtype="int64"),
        np.array(pool_key, dtype="int64").reshape(-1, 2),
        runs[:, 0], runs[:, 1], runs[:, 2], runs[:, 3],
    )


with pm.Model() as model:
    rule_count = pm.Data("rule_count", np.zeros((1, N_RULES)))  # (sets,): examples in each rule
    n_examples = pm.Data("n_examples", np.zeros(1))  # (sets,)
    # Set pools: intervals of one size containing the same number of the set's examples.
    set_run_pool = pm.Data("set_run_pool", np.zeros(1, dtype="int64"))
    set_run_size = pm.Data("set_run_size", np.zeros(1, dtype="int64"))
    set_run_first = pm.Data("set_run_first", np.zeros(1, dtype="int64"))
    set_run_end = pm.Data("set_run_end", np.zeros(1, dtype="int64"))
    set_pool_set = pm.Data("set_pool_set", np.zeros(1, dtype="int64"))
    set_pool_size = pm.Data("set_pool_size", np.zeros(1, dtype="int64"))
    set_pool_count = pm.Data("set_pool_count", np.zeros(1))
    # Cell pools: the same, restricted to intervals containing the cell's target.
    cell_run_pool = pm.Data("cell_run_pool", np.zeros(1, dtype="int64"))
    cell_run_size = pm.Data("cell_run_size", np.zeros(1, dtype="int64"))
    cell_run_first = pm.Data("cell_run_first", np.zeros(1, dtype="int64"))
    cell_run_end = pm.Data("cell_run_end", np.zeros(1, dtype="int64"))
    cell_pool_cell = pm.Data("cell_pool_cell", np.zeros(1, dtype="int64"))
    cell_pool_size = pm.Data("cell_pool_size", np.zeros(1, dtype="int64"))
    cell_pool_count = pm.Data("cell_pool_count", np.zeros(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # (cells,): target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (responses,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    # Size-principle strength: 0 = weak sampling, 1 = full strong sampling.
    alpha = pm.Beta("alpha", 2, 2)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    # Prior of one interval at each grid point, normalised over the intervals.
    point_prior = (1 - p_mix) * pt.exp(
        log_density - pt.logsumexp(log_density[INTERVAL_GRID_POINT])
    )  # (199,)
    # cumulative_prior[L, m]: prior mass of the intervals of size index L
    # starting before m.
    padded = pt.concatenate([point_prior, pt.zeros(1)])
    cumulative_prior = pt.concatenate(
        [pt.zeros((100, 1)), pt.cumsum(padded[SIZE_START_POINT], axis=1)], axis=1
    )  # (100, 102)

    # log P(x | h) for an example inside / outside a concept of each size.
    log_out = np.log(EPSILON) - alpha * np.log(100.0)
    log_in = pt.log((1 - EPSILON) * pt.exp(-alpha * LOG_SIZES) + EPSILON * pt.exp(-alpha * np.log(100.0)))

    def pooled_lik(size, count, n):
        return pt.exp(count * log_in[size] + (n - count) * log_out)

    rule_lik = p_mix * rule_weights[None, :] * pt.exp(
        rule_count * log_in[RULE_SIZE_INDEX][None, :] + (n_examples[:, None] - rule_count) * log_out
    )  # (sets, rules): prior x likelihood

    set_mass = pt.zeros_like(set_pool_count)[set_run_pool].inc(
        cumulative_prior[set_run_size, set_run_end] - cumulative_prior[set_run_size, set_run_first]
    )
    set_terms = set_mass * pooled_lik(set_pool_size, set_pool_count, n_examples[set_pool_set])
    evidence = rule_lik.sum(axis=1) + pt.zeros_like(n_examples)[set_pool_set].inc(set_terms)

    cell_mass = pt.zeros_like(cell_pool_count)[cell_run_pool].inc(
        cumulative_prior[cell_run_size, cell_run_end] - cumulative_prior[cell_run_size, cell_run_first]
    )
    cell_terms = cell_mass * pooled_lik(
        cell_pool_size, cell_pool_count, n_examples[cell_set][cell_pool_cell]
    )
    cell_rules = (rule_lik[cell_set] * pt.as_tensor_variable(RULE_MEMBERSHIP.T)[cell_target]).sum(axis=1)
    numerator = cell_rules + pt.zeros_like(cell_set, dtype="float64")[cell_pool_cell].inc(cell_terms)
    in_concept = numerator / evidence[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    counts = np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_counts = counts[:, N_RULES:].astype("int64")
    set_keys = [np.stack([INTERVAL_SIZE_INDEX, c], axis=1) for c in interval_counts]
    cell_set = np.array([index[s] for s, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells], dtype="int64")
    cell_keys = []
    for s, t in zip(cell_set, cell_target):
        keys = set_keys[s].copy()
        keys[INTERVAL_MEMBERSHIP[:, t] == 0] = -1
        cell_keys.append(keys)
    set_row, set_key, set_pool, set_size, set_first, set_end = _pooling(set_keys)
    cell_row, cell_key, cell_pool, cell_size, cell_first, cell_end = _pooling(cell_keys)
    return {
        "rule_count": counts[:, :N_RULES],
        "n_examples": np.array([float(len(s)) for s in unique]),
        "set_run_pool": set_pool,
        "set_run_size": set_size,
        "set_run_first": set_first,
        "set_run_end": set_end,
        "set_pool_set": set_row,
        "set_pool_size": set_key[:, 0],
        "set_pool_count": set_key[:, 1].astype("float64"),
        "cell_run_pool": cell_pool,
        "cell_run_size": cell_size,
        "cell_run_first": cell_first,
        "cell_run_end": cell_end,
        "cell_pool_cell": cell_row,
        "cell_pool_size": cell_key[:, 0],
        "cell_pool_count": cell_key[:, 1].astype("float64"),
        "cell_set": cell_set,
        "cell_target": cell_target,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
