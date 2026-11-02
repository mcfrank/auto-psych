"""Surprise-weighted sequential updating in family-prior concept learning.

People read the examples one at a time, revising a running Bayesian guess about
the concept (45 rules with separate prior plausibilities for even, odd and the
other kinds of rule, plus all intervals of 1-100), and each new example's
evidence is weighted by how surprising it is under the guess formed from the
examples before it: weight_k = beta * exp(kappa * (surprise_k - log 100)), where
surprise_k = -log of the running posterior-predictive probability of example k.
kappa > 0: pattern-violating examples count more; kappa = 0 recovers an
order-blind tempered learner. The first example is judged against the prior.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_N = 4


def rule_concepts():
    """(family, members) for each rule; families index FAMILIES."""
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    rules = [
        (0, [n for n in NUMBERS if n % 2 == 1]),
        (1, list(range(2, 101, 2))),
        (2, primes),
        (3, [n * n for n in range(1, 11)]),
        (3, [n**3 for n in range(1, 5)]),
        (4, [2, 4, 8, 16, 64]),
        (4, [2, 4, 8, 16, 32, 37, 64]),
    ]
    rules += [(5, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [(4, [m**k for k in range(1, 8) if m**k <= 100]) for m in range(2, 11)]
    rules += [(6, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    rules += [(7, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return rules


FAMILIES = ["odd", "even", "primes", "squares_cubes", "powers", "multiples", "final_digit", "contains_digit"]
_RULES = rule_concepts()
RULE_FAMILY = np.array([f for f, _ in _RULES], dtype="int64")
N_RULES = len(_RULES)
N_FAM = len(FAMILIES)
FAMILY_LOG_COUNT = np.log(np.bincount(RULE_FAMILY, minlength=N_FAM)).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in _RULES], dtype="float64")
INTERVALS = np.array([(m, n) for m in NUMBERS for n in range(m, 101)])
N_INTERVALS = len(INTERVALS)
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= INTERVALS[:, :1]) & (NUMBERS[None, :] <= INTERVALS[:, 1:])).astype("float64")
MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])
EX_LL = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)

with pm.Model() as model:
    ex_ll = pm.Data("ex_ll", np.zeros((1, MAX_N, N_RULES + 1)))  # (sets, position, rules + interval groups)
    ex_mask = pm.Data("ex_mask", np.zeros((1, MAX_N)))  # 1 where the position holds an example
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, interval groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # set of each distinct (set, target) pair
    cell_in = pm.Data("cell_in", np.zeros((1, N_RULES + 1)))  # (pairs, hyps): target's membership
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # pair index of each trial
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # baseline sampling strength per example
    kappa = pm.Normal("kappa", mu=0.0, sigma=0.5)  # surprise sensitivity of an example's weight

    log_norm = pt.logsumexp(pt.concatenate([family_logit, pt.zeros(1)]))
    log_rule_prior = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] - log_norm
    log_groups_prior = group_log_size - log_norm - np.log(N_INTERVALS)
    log_rules_prior = pt.zeros((group_log_size.shape[0], 1)) + log_rule_prior[None, :]
    log_joint = pt.concatenate([log_rules_prior, log_groups_prior], axis=1)  # (sets, hyps)

    for k in range(MAX_N):
        ll_k = ex_ll[:, k, :]
        log_post = log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True)
        surprise = -pt.logsumexp(log_post + ll_k, axis=1)  # -log running predictive of example k
        weight = beta * pt.exp(kappa * pt.clip(surprise - np.log(100.0), -6.0, 6.0)) * ex_mask[:, k]
        log_joint = log_joint + weight[:, None] * ll_k

    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_in).sum(axis=1)[row_cell]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Per distinct set (in presented order): each example's log-likelihood under every rule and
    interval group. Intervals with identical log-likelihood at every position share every running
    posterior, so they are pooled; padding groups get log size -1e4 (no mass)."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    interval_ll = EX_LL[N_RULES:]
    groupings = []
    for s in unique:
        keys, inverse = np.unique(np.round(interval_ll[:, np.array(s) - 1], 9), axis=0, return_inverse=True)
        groupings.append((inverse.reshape(-1), keys))
    n_groups = max(len(k) for _, k in groupings)
    ll = np.zeros((len(unique), MAX_N, N_RULES + n_groups))
    mask = np.zeros((len(unique), MAX_N))
    log_size = np.full((len(unique), n_groups), -1e4)
    sizes = np.ones((len(unique), n_groups))
    for i, (s, (inverse, keys)) in enumerate(zip(unique, groupings)):
        counts = np.bincount(inverse, minlength=len(keys))
        log_size[i, : len(keys)], sizes[i, : len(keys)] = np.log(counts), counts
        for k, x in enumerate(s[:MAX_N]):
            ll[i, k, :N_RULES] = EX_LL[:N_RULES, x - 1]
            ll[i, k, N_RULES : N_RULES + len(keys)] = keys[:, k]
            mask[i, k] = 1.0
    cells = [(set_index[s], int(float(r["target"]))) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: j for j, c in enumerate(unique_cells)}
    cell_in = np.zeros((len(unique_cells), N_RULES + n_groups))
    for j, (i, t) in enumerate(unique_cells):
        inverse, keys = groupings[i]
        cell_in[j, :N_RULES] = RULE_MEMBERSHIP[:, t - 1]
        frac = np.bincount(inverse, weights=INTERVAL_MEMBERSHIP[:, t - 1], minlength=len(keys)) / sizes[i, : len(keys)]
        cell_in[j, N_RULES : N_RULES + len(keys)] = frac
    return {
        "ex_ll": ll,
        "ex_mask": mask,
        "group_log_size": log_size,
        "cell_set": np.array([i for i, _ in unique_cells], dtype="int64"),
        "cell_in": cell_in,
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
