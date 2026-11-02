"""Refinement of learned_rule_prior_mixture: Bayesian concept learning over 45
rules (each with its own learned prior plausibility) and all intervals of
1..100, but the size principle is applied only partially: the examples'
likelihood is tempered by a free sampling-strength exponent beta (1 = strong
sampling, 0 = weak sampling), as in tempered_size_principle_mixture. "Yes" is
the posterior probability that the concept contains the target. Single change
versus the incumbent: full -> tempered example likelihood.
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
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    igroup_log_lik = pm.Data("igroup_log_lik", np.zeros((1, 1)))  # (sets, interval groups)
    igroup_log_size = pm.Data("igroup_log_size", np.zeros((1, 1)))  # (sets, interval groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_ifrac = pm.Data("cell_ifrac", np.zeros((1, 1)))  # (cells, interval groups)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    rule_logit = pm.ZeroSumNormal("rule_logit", sigma=1.0, shape=N_RULES)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength

    log_rules = pt.log(lambda_0) + pt.special.log_softmax(rule_logit)[None, :] + beta * rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + igroup_log_size + beta * igroup_log_lik
    log_joint = pt.concatenate([log_rules, log_intervals], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    post_cells = posterior[cell_set]
    in_concept = (post_cells[:, :N_RULES] * cell_rule_in).sum(axis=1) + (post_cells[:, N_RULES:] * cell_ifrac).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`.
    Intervals with identical example likelihood are pooled into groups (their
    tempered posterior weights stay proportional to group size)."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: i for i, c in enumerate(unique_cells)}

    rule_ll = np.zeros((len(unique), N_RULES))
    groupings = []
    for i, s in enumerate(unique):
        ll = EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        rule_ll[i] = ll[:N_RULES]
        keys, inverse = np.unique(np.round(ll[N_RULES:], 9), return_inverse=True)
        groupings.append((inverse.reshape(-1), keys))
    n_groups = max(len(k) for _, k in groupings)
    g_ll = np.zeros((len(unique), n_groups))
    g_size = np.full((len(unique), n_groups), -1e4)
    sizes = np.ones((len(unique), n_groups))
    for i, (inverse, keys) in enumerate(groupings):
        counts = np.bincount(inverse, minlength=len(keys))
        g_ll[i, : len(keys)] = keys
        g_size[i, : len(keys)], sizes[i, : len(keys)] = np.log(counts), counts
    frac = np.zeros((len(unique_cells), n_groups))
    rule_in = np.zeros((len(unique_cells), N_RULES))
    for j, (i, t) in enumerate(unique_cells):
        inverse, keys = groupings[i]
        frac[j, : len(keys)] = np.bincount(inverse, weights=INTERVAL_MEMBERSHIP[:, t - 1], minlength=len(keys))
        frac[j] /= sizes[i]
        rule_in[j] = RULE_MEMBERSHIP[:, t - 1]
    return {
        "rule_log_lik": rule_ll,
        "igroup_log_lik": g_ll,
        "igroup_log_size": g_size,
        "cell_set": np.array([c[0] for c in unique_cells], dtype="int64"),
        "cell_rule_in": rule_in,
        "cell_ifrac": frac,
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
