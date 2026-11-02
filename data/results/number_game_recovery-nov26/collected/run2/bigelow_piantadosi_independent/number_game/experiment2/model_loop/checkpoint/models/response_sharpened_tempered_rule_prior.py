"""Refinement of tempered_learned_rule_prior: Bayesian concept learning over 45
rules (each with its own learned prior plausibility) and all intervals of
1..100 with a tempered size principle (exponent beta). The belief that the
target is in the concept is mapped to a yes/no answer through a decisiveness
exponent gamma on its log-odds, p_yes = sigmoid(gamma * logit(belief)), so
people answer more consistently (gamma > 1) than probability matching. Single
change: this sharpened belief-to-response mapping.
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
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, 1)))  # (sets, interval groups)
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, interval groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # (cells,) target - 1
    cell_frac = pm.Data("cell_frac", np.zeros((1, 1)))  # (cells, groups) share of group containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    rule_logit = pm.ZeroSumNormal("rule_logit", sigma=1.0, shape=N_RULES)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength
    gamma = pm.LogNormal("gamma", mu=0.0, sigma=0.5)  # response decisiveness
    log_rule_prior = pt.log(lambda_0) + pt.special.log_softmax(rule_logit)

    log_rules = log_rule_prior[None, :] + beta * rule_log_lik
    log_groups = pt.log1p(-lambda_0) - 2 * np.log(100) + group_log_size + beta * group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    post_cells = posterior[cell_set]
    rule_in = (post_cells[:, :N_RULES] * pt.as_tensor(RULE_MEMBERSHIP).T[cell_target]).sum(axis=1)
    interval_in = (post_cells[:, N_RULES:] * cell_frac).sum(axis=1)
    in_concept = rule_in + interval_in

    belief = pt.clip(in_concept, 1e-9, 1 - 1e-9)
    decided = pm.math.sigmoid(gamma * (pt.log(belief) - pt.log1p(-belief)))
    p_yes = pm.Deterministic("p_yes", pt.clip(decided[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`.

    Intervals with identical example likelihood are pooled into groups (their
    posterior weights move together for any beta); per (set, target) cell we
    record the fraction of each group's intervals that contain the target.
    """
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: i for i, c in enumerate(unique_cells)}

    set_ll = [EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]
    groupings = [np.unique(np.round(ll[N_RULES:], 9), return_inverse=True) for ll in set_ll]
    n_groups = max(len(k) for k, _ in groupings)
    log_lik = np.zeros((len(unique), n_groups))
    log_size = np.full((len(unique), n_groups), -1e4)
    frac_by_set = []
    for i, (keys, inverse) in enumerate(groupings):
        inverse = inverse.reshape(-1)
        counts = np.bincount(inverse, minlength=n_groups).astype("float64")
        log_lik[i, : len(keys)] = keys
        log_size[i, : len(keys)] = np.log(counts[: len(keys)])
        onehot = np.zeros((len(inverse), n_groups))
        onehot[np.arange(len(inverse)), inverse] = 1.0
        frac_by_set.append((onehot.T @ INTERVAL_MEMBERSHIP) / np.maximum(counts, 1.0)[:, None])  # (groups, 100)
    cell_frac = np.stack([frac_by_set[si][:, t - 1] for si, t in unique_cells])
    return {
        "rule_log_lik": np.stack([ll[:N_RULES] for ll in set_ll]),
        "group_log_lik": log_lik,
        "group_log_size": log_size,
        "cell_set": np.array([si for si, _ in unique_cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in unique_cells], dtype="int64"),
        "cell_frac": cell_frac,
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
