"""Refinement of erlang_interval_prior_concepts: Bayesian concept learning over
the same 45 rules and all 5,050 intervals of 1..100 (size principle, examples
drawn from the concept with probability 0.9 and uniformly otherwise, a free
prior weight per rule and a rule share lambda_0, an Erlang prior on interval
size with fitted characteristic size sigma), with one change: people favour
ranges with round-number boundaries. An interval [m, n] gets an extra prior
factor exp(beta * r), where r counts its round endpoints (m = 1 or m a
multiple of 10; n a multiple of 10) and beta is fitted, so "20 to 40" is a
priori more plausible than "17 to 43" of similar width. "Yes" is the posterior
probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
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


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE = INTERVAL_MEMBERSHIP.sum(axis=1).astype(int)
_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INTERVAL_ROUND = (((_LO == 1) | (_LO % 10 == 0)).astype(int) + (_HI % 10 == 0).astype(int))

# An interval's likelihood depends only on its size s and on how many examples
# k it contains, and its prior only on s and its round-endpoint count r, so for
# each example set the 5,050 intervals are grouped by (s, k, r). Per set,
# prepare_observed keeps only the non-empty groups (padded to a common width),
# storing each group's s, k, r and log count, and, for each (set, target) cell
# asked about, the share of the group's intervals containing the target; the
# posterior below is exactly the one over all intervals.
EMPTY_LOG_COUNT = -1e4

# As in the incumbent: "ends in 0" and "contains digit 0" are one set, so
# their weights are sampled as a shared Dirichlet component and a Beta split.
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0
IN_LOG = np.log(ALPHA / np.arange(1, 101) + (1 - ALPHA) / 100)  # by interval size
OUT_LOG = np.log((1 - ALPHA) / 100)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_size = pm.Data("group_size", np.ones((1, 1)))  # (sets, slots)
    group_round = pm.Data("group_round", np.zeros((1, 1)))  # (sets, slots)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, 1)))  # (sets, slots), examples given the interval
    group_log_count = pm.Data("group_log_count", np.zeros((1, 1)))  # (sets, slots)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_group_share = pm.Data("cell_group_share", np.zeros((1, 1)))  # (cells, slots)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    # Erlang prior on interval size with characteristic size sigma (1..100 scale),
    # times exp(round_boost) per round endpoint (1 or a multiple of 10 below,
    # a multiple of 10 above) -- the one change from the incumbent.
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
    round_boost = pm.Normal("round_boost", mu=0.0, sigma=1.5)
    size = pt.as_tensor(INTERVAL_SIZE.astype("float64"))
    log_norm = pt.logsumexp(pt.log(size) - size / sigma + round_boost * pt.as_tensor(INTERVAL_ROUND.astype("float64")))
    group_log_prior = pt.log(group_size) - group_size / sigma + round_boost * group_round - log_norm

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) + group_log_prior + group_log_count + group_log_lik
    log_z = pt.logaddexp(pt.logsumexp(log_rules, axis=1), pt.logsumexp(log_intervals, axis=1))
    post_rules = pt.exp(log_rules - log_z[:, None])
    post_intervals = pt.exp(log_intervals - log_z[:, None])
    in_concept = ((post_rules[cell_set] * cell_rule_member).sum(axis=1)
                  + (post_intervals[cell_set] * cell_group_share).sum(axis=1))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    n_groups = 100 * 5 * 3
    per_set = []
    for s in unique:
        k = sum(((_LO <= x) & (_HI >= x)).astype(int) for x in s)
        group = ((INTERVAL_SIZE - 1) * 5 + k) * 3 + INTERVAL_ROUND
        count = np.bincount(group, minlength=n_groups).astype("float64")
        members = np.zeros((n_groups, 100))
        np.add.at(members, group, INTERVAL_MEMBERSHIP)
        filled = np.flatnonzero(count > 0)
        g_size = filled // 15 + 1
        g_k = (filled // 3) % 5
        g_lik = g_k * IN_LOG[g_size - 1] + (len(s) - g_k) * OUT_LOG
        per_set.append((g_size, filled % 3, g_lik, np.log(count[filled]), members[filled] / count[filled, None]))
    width = max(len(p[0]) for p in per_set)
    n_sets = len(unique)
    g_size = np.ones((n_sets, width))
    g_round = np.zeros((n_sets, width))
    g_lik = np.zeros((n_sets, width))
    g_count = np.full((n_sets, width), EMPTY_LOG_COUNT)
    for i, (sz, rd, lk, lc, _) in enumerate(per_set):
        n = len(sz)
        g_size[i, :n], g_round[i, :n], g_lik[i, :n], g_count[i, :n] = sz, rd, lk, lc
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    share = np.zeros((len(cells), width))
    for j, (i, t) in enumerate(cells):
        sh = per_set[i][4][:, t - 1]
        share[j, :len(sh)] = sh
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "group_size": g_size,
        "group_round": g_round,
        "group_log_lik": g_lik,
        "group_log_count": g_count,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_group_share": share,
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
