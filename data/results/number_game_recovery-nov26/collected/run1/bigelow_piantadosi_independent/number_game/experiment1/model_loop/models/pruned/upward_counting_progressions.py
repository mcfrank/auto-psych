"""Tempered, exception-tolerant Bayesian concept learning with counting-up progressions.

People do exception-tolerant, tempered Bayesian concept learning over the 45
arithmetic rules and all intervals of 1..100, but their concept vocabulary
also contains counting-up sequences: arithmetic progressions {a, a+d, a+2d, ...
<= 100} (start a, step d = 2..50), the whole family sharing a fitted share of
the prior equally. Because such a sequence starts at or below the smallest
example and runs upward, people fill in the examples' range at the inferred
spacing and generalise above the examples more than below them.
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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Counting-up progressions: start a, step d, every a + k*d <= 100 (at least 2 members).
PROGRESSIONS = [(a, d) for d in range(2, 51) for a in range(1, 101 - d)]
N_PROG = len(PROGRESSIONS)
PROG_MEMBERSHIP = np.zeros((N_PROG, 100))
for _j, (_a, _d) in enumerate(PROGRESSIONS):
    PROG_MEMBERSHIP[_j, np.arange(_a, 101, _d) - 1] = 1.0
PROG_SIZE = PROG_MEMBERSHIP.sum(axis=1)
# Progressions grouped by (size, examples inside) per set -- the likelihood depends
# only on these, so the grouping is exact (as for intervals).
MAX_PROG_SIZE = int(PROG_SIZE.max())
N_PGROUPS = MAX_PROG_SIZE * 5
PGROUP_SIZE = np.repeat(np.arange(1, MAX_PROG_SIZE + 1), 5).astype("float64")
PGROUP_K = np.tile(np.arange(5), MAX_PROG_SIZE).astype("float64")

MAX_EXAMPLES = 4
SIZES = np.arange(1, 101)
N_GROUPS = 100 * (MAX_EXAMPLES + 1)
GROUP_SIZE = np.repeat(SIZES, MAX_EXAMPLES + 1).astype("float64")
GROUP_K = np.tile(np.arange(MAX_EXAMPLES + 1), 100).astype("float64")

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))
    pgroup_log_count = pm.Data("pgroup_log_count", np.zeros((1, N_PGROUPS)))
    set_n = pm.Data("set_n", np.ones(1))
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_pgroup_in_concept = pm.Data("cell_pgroup_in_concept", np.zeros((1, N_PGROUPS)))
    cell_group_in_concept = pm.Data("cell_group_in_concept", np.zeros((1, N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior shares of rules, progressions and intervals.
    w = pm.Dirichlet("family_weights", a=np.array([2.0, 2.0, 2.0]))
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples

    log_noise = pt.log(eps) - np.log(100)
    n_col = set_n[:, None]

    def family_log_lik(k, size):
        log_in = pt.log((1 - eps) / size + eps / 100)
        return k * log_in[None, :] + (n_col - k) * log_noise

    log_rules = pt.log(w[0]) - np.log(N_RULES) + beta * family_log_lik(rule_k, RULE_SIZE)
    log_progs = (
        pt.log(w[1]) - np.log(N_PROG) + beta * family_log_lik(PGROUP_K[None, :], PGROUP_SIZE) + pgroup_log_count
    )
    log_groups = (
        pt.log(w[2]) - 2 * np.log(100) + beta * family_log_lik(GROUP_K[None, :], GROUP_SIZE) + group_log_count
    )
    log_joint = pt.concatenate([log_rules, log_progs, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))[cell_set]
    in_concept = (
        (posterior[:, :N_RULES] * cell_rule_member).sum(axis=1)
        + (posterior[:, N_RULES : N_RULES + N_PGROUPS] * cell_pgroup_in_concept).sum(axis=1)
        + (posterior[:, N_RULES + N_PGROUPS :] * cell_group_in_concept).sum(axis=1)
    )

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _interval_groups(examples):
    count = np.zeros(N_GROUPS)
    in_concept = np.zeros((N_GROUPS, 100))
    ex = np.array(examples)
    for s in SIZES:
        for m in range(1, 102 - s):
            k = int(((ex >= m) & (ex <= m + s - 1)).sum())
            g = (s - 1) * (MAX_EXAMPLES + 1) + k
            count[g] += 1
            in_concept[g, m - 1 : m + s - 1] += 1
    nonzero = count > 0
    in_concept[nonzero] /= count[nonzero, None]
    log_count = np.where(nonzero, np.log(np.maximum(count, 1)), -1e4)
    return log_count, in_concept


def _progression_groups(examples):
    k = PROG_MEMBERSHIP[:, np.array(examples) - 1].sum(axis=1).astype(int)
    g = (PROG_SIZE.astype(int) - 1) * 5 + k
    count = np.bincount(g, minlength=N_PGROUPS).astype("float64")
    in_concept = np.zeros((N_PGROUPS, 100))
    np.add.at(in_concept, g, PROG_MEMBERSHIP)
    nonzero = count > 0
    in_concept[nonzero] /= count[nonzero, None]
    log_count = np.where(nonzero, np.log(np.maximum(count, 1)), -1e4)
    return log_count, in_concept


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    groups = [_interval_groups(s) for s in unique]
    pgroups = [_progression_groups(s) for s in unique]
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "pgroup_log_count": np.stack([g[0] for g in pgroups]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "group_log_count": np.stack([g[0] for g in groups]),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_pgroup_in_concept": np.stack([pgroups[i][1][:, t - 1] for i, t in cells]),
        "cell_group_in_concept": np.stack([groups[i][1][:, t - 1] for i, t in cells]),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
