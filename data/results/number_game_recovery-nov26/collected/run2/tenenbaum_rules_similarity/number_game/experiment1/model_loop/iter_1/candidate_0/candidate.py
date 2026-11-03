"""Bayesian concept learning with a fading memory for earlier examples.

People do Bayesian concept learning over the 45 rules and all intervals of
1-100 (independently weighted rule prior, lambda_0 share for rules, size
principle with 10% random examples), but their memory of the examples fades as
they read the set: the most recently shown example counts in full, and each
earlier example's log-evidence is discounted by a factor delta per later
example. delta = 1 recovers the incumbent; delta < 1 lets concepts that miss
early examples survive, so generalisation follows the end of the list.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_LEN = 4


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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EXAMPLE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Each interval has prior (1 - lambda_0) / 100^2. Within one example set, all
# intervals of the same length that contain the same subset of the examples
# have the same likelihood for every delta, so prepare_observed groups them:
# per group its log size, the per-position log-likelihood of one interval, and
# the fraction of its intervals containing each target. This is exact.
INTERVAL_LOG_PRIOR = -2 * np.log(100)

with pm.Model() as model:
    # Positions ordered most recent first, zero-padded for shorter sets.
    rule_pos_log_lik = pm.Data("rule_pos_log_lik", np.zeros((1, MAX_LEN, N_RULES)))
    group_pos_log_lik = pm.Data("group_pos_log_lik", np.zeros((1, MAX_LEN, 1)))  # (sets, positions, groups)
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, groups); padding very negative
    # Only the observed (set, target) cells are evaluated.
    cell_set = pm.Data("cell_set", np.zeros((1, 1)))  # (cells, sets): one-hot, the cell's set
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))  # (cells, rules): rule contains target
    cell_group_in = pm.Data("cell_group_in", np.zeros((1, 1)))  # (cells, groups): share containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,): the trial's cell
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    # Retention per later example: an example k positions before the last counts delta^k.
    delta = pm.Beta("delta", 4, 1.5)
    decay = delta ** pt.arange(MAX_LEN)

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + (rule_pos_log_lik * decay[None, :, None]).sum(axis=1)
    log_groups = (pt.log1p(-lambda_0) + INTERVAL_LOG_PRIOR + group_log_size
                  + (group_pos_log_lik * decay[None, :, None]).sum(axis=1))
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_posterior = cell_set @ posterior  # (cells, rules + groups); a matmul, not a slow gather gradient
    in_concept = (cell_posterior[:, :N_RULES] * cell_rule_in).sum(axis=1) + (cell_posterior[:, N_RULES:] * cell_group_in).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _interval_groups(examples):
    """Group the intervals of 1..100 by (length, which examples they contain)."""
    groups = {}
    for m in range(1, 101):
        for n in range(m, 101):
            key = (n - m + 1, tuple(m <= x <= n for x in examples))
            g = groups.setdefault(key, [0, np.zeros(100)])
            g[0] += 1
            g[1][m - 1:n] += 1
    return groups


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    grouped = [_interval_groups(s) for s in unique]
    n_groups = max(len(g) for g in grouped)
    rule_pos = np.zeros((len(unique), MAX_LEN, N_RULES))
    group_pos = np.zeros((len(unique), MAX_LEN, n_groups))
    log_size = np.full((len(unique), n_groups), -1e4)
    in_concept = np.zeros((len(unique), n_groups, 100))
    for i, (s, groups) in enumerate(zip(unique, grouped)):
        recent_first = list(reversed(s))
        for k, x in enumerate(recent_first):
            rule_pos[i, k] = RULE_EXAMPLE_LOG_LIK[:, x - 1]
        for j, ((length, contains), (count, covered)) in enumerate(groups.items()):
            log_size[i, j] = np.log(count)
            in_concept[i, j] = covered / count
            for k, inside in enumerate(reversed(contains)):
                group_pos[i, k, j] = np.log(ALPHA * inside / length + (1 - ALPHA) / 100)
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "rule_pos_log_lik": rule_pos,
        "group_pos_log_lik": group_pos,
        "group_log_size": log_size,
        "cell_set": np.eye(len(unique))[[i for i, _ in cells]],
        "cell_rule_in": np.array([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_group_in": np.array([in_concept[i, :, t - 1] for i, t in cells]),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
