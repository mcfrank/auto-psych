"""Bigelow & Piantadosi's Mixture Model with a tempered size principle.

Refinement of `bigelow_piantadosi_mixture`. The learner considers the same 45
rule concepts and the 5,050 intervals of 1..100, with prior lambda_0 split
equally over the rules and (1 - lambda_0) / 100^2 on each interval; each
example comes from the concept with probability 0.9 (uniformly) and from
1..100 otherwise. The one change: people only partially trust that examples
were sampled from the concept, so the log-likelihood of the example set is
scaled by a sampling-strength exponent beta (beta = 1 is the incumbent's
strong sampling, beta -> 0 is weak sampling). "Yes" is the posterior
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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
N_HYP = N_RULES + len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
# log P(x | h) of one example x (column x - 1) under each hypothesis.
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
IS_RULE = np.r_[np.ones(N_RULES), np.zeros(len(INTERVALS))]

# Hypotheses with the same prior and the same likelihood of a set have the
# same posterior at every parameter value, so prepare_observed groups them per
# distinct set (at most a few hundred groups instead of 5,095): each group's
# log size, whether it holds rules, log P(D | h), and per (set, target) cell
# the fraction of the group's hypotheses that contain the target. Padding
# groups get a log size of -1e4, i.e. no posterior mass.

with pm.Model() as model:
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, 1)))  # (sets, groups)
    group_is_rule = pm.Data("group_is_rule", np.zeros((1, 1)))  # (sets, groups)
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,) set index
    cell_frac = pm.Data("cell_frac", np.zeros((1, 1)))  # (cells, groups)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (rows,) cell index
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength

    log_prior = group_is_rule * (pt.log(lambda_0) - np.log(N_RULES)) + (1 - group_is_rule) * (
        pt.log1p(-lambda_0) - 2 * np.log(100)
    )
    log_joint = log_prior + group_log_size + beta * group_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_frac).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: i for i, c in enumerate(unique_cells)}

    groupings = []  # per set: (group id of each hypothesis, group keys)
    for s in unique:
        ll = EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        keys, inverse = np.unique(np.c_[IS_RULE, np.round(ll, 9)], axis=0, return_inverse=True)
        groupings.append((inverse.reshape(-1), keys))
    n_groups = max(len(k) for _, k in groupings)
    log_lik = np.zeros((len(unique), n_groups))
    is_rule = np.zeros((len(unique), n_groups))
    log_size = np.full((len(unique), n_groups), -1e4)
    sizes = np.ones((len(unique), n_groups))
    for i, (inverse, keys) in enumerate(groupings):
        counts = np.bincount(inverse, minlength=len(keys))
        is_rule[i, : len(keys)], log_lik[i, : len(keys)] = keys[:, 0], keys[:, 1]
        log_size[i, : len(keys)], sizes[i, : len(keys)] = np.log(counts), counts
    frac = np.zeros((len(unique_cells), n_groups))
    for j, (i, t) in enumerate(unique_cells):
        inverse = groupings[i][0]
        frac[j, : len(groupings[i][1])] = np.bincount(inverse, weights=MEMBERSHIP[:, t - 1], minlength=len(groupings[i][1]))
        frac[j] /= sizes[i]
    return {
        "group_log_lik": log_lik,
        "group_is_rule": is_rule,
        "group_log_size": log_size,
        "cell_set": np.array([c[0] for c in unique_cells], dtype="int64"),
        "cell_frac": frac,
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
