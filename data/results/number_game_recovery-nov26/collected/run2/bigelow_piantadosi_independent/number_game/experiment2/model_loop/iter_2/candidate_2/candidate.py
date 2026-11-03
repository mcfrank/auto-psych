"""Noisy target-magnitude membership in capacity-limited concept learning.

People infer the hidden concept by capacity-limited Bayesian learning over 45
rules (separate prior plausibilities for even, odd and the other kinds of rule)
and all intervals of 1-100 (evidence scaled by beta * n ** (-gamma)), but they
check the target against the concept with a noisy sense of its magnitude: the
target is encoded as a nearby number t + d with probability proportional to
exp(-|d| / width) (truncated to |d| <= 6 and to 1-100), and the answer is yes
with the probability that the encoded number is in the concept. Targets that
break a well-supported rule but lie next to its members thus get a graded share
of yeses; width -> 0 recovers exact membership.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_SHIFT = 6
DISTANCES = np.arange(MAX_SHIFT + 1)  # |shift| of the encoded number from the target


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
INTERVAL_MEMBERSHIP = (NUMBERS[None, :] >= INTERVALS[:, :1]) & (NUMBERS[None, :] <= INTERVALS[:, 1:])
INTERVAL_MEMBERSHIP = INTERVAL_MEMBERSHIP.astype("float64")


def _example_log_lik(membership):
    return np.log(ALPHA * membership / membership.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


RULE_EX_LL = _example_log_lik(RULE_MEMBERSHIP)  # (rules, 100)
INTERVAL_EX_LL = _example_log_lik(INTERVAL_MEMBERSHIP)  # (intervals, 100)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, 1)))  # (sets, interval groups)
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, interval groups)
    set_log_n = pm.Data("set_log_n", np.zeros(1))  # log(number of examples) per set
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,) distinct (set, target)
    cell_member = pm.Data("cell_member", np.zeros((1, len(DISTANCES), N_RULES + 1)))  # (cells, distances, concepts)
    cell_valid = pm.Data("cell_valid", np.ones((1, len(DISTANCES))))  # numbers at that distance within 1-100
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (rows,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength of a single example
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)  # capacity limit: loss of evidence per (log) example
    width = pm.LogNormal("width", mu=np.log(0.7), sigma=0.8)  # magnitude noise of the target

    log_norm = pt.logsumexp(pt.concatenate([family_logit, pt.zeros(1)]))
    log_rule_prior = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS)

    beta_set = (beta * pt.exp(-gamma * set_log_n))[:, None]
    log_rules = log_rule_prior[None, :] + beta_set * rule_log_lik
    log_groups = log_interval_prior + group_log_size + beta_set * group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    post_cell = posterior[cell_set]  # (cells, concepts)
    in_dist = pt.batched_dot(cell_member, post_cell)  # (cells, distances): summed over t - d and t + d

    kernel = pt.exp(-DISTANCES.astype("float64") / width)[None, :] * cell_valid
    kernel = kernel / kernel.sum(axis=1, keepdims=True)
    in_concept = (kernel * in_dist).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`.

    Intervals with the same likelihood of a set share a posterior and are grouped
    per distinct set; padding groups get a log size of -1e4 (no posterior mass).
    For each distinct (set, target) cell, the membership of every number within
    MAX_SHIFT of the target in each rule and interval group is precomputed.
    """
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}

    rule_ll = np.stack([RULE_EX_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    groupings = []
    for s in unique:
        ll = INTERVAL_EX_LL[:, np.array(s) - 1].sum(axis=1)
        keys, inverse = np.unique(np.round(ll, 9), return_inverse=True)
        groupings.append((inverse.reshape(-1), keys))
    n_groups = max(len(k) for _, k in groupings)
    log_lik = np.zeros((len(unique), n_groups))
    log_size = np.full((len(unique), n_groups), -1e4)
    group_frac = []  # per set: (groups, 100) fraction of the group's intervals containing each number
    for i, (inverse, keys) in enumerate(groupings):
        counts = np.bincount(inverse, minlength=len(keys))
        log_lik[i, : len(keys)] = keys
        log_size[i, : len(keys)] = np.log(counts)
        frac = np.zeros((n_groups, 100))
        np.add.at(frac, inverse, INTERVAL_MEMBERSHIP)
        frac[: len(keys)] /= counts[:, None]
        group_frac.append(frac)

    cells = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: k for k, c in enumerate(unique_cells)}
    member = np.zeros((len(unique_cells), len(DISTANCES), N_RULES + n_groups))
    valid = np.zeros((len(unique_cells), len(DISTANCES)))
    for k, (i, t) in enumerate(unique_cells):
        for d in range(-MAX_SHIFT, MAX_SHIFT + 1):
            u = t + d
            if 1 <= u <= 100:
                valid[k, abs(d)] += 1.0
                member[k, abs(d), :N_RULES] += RULE_MEMBERSHIP[:, u - 1]
                member[k, abs(d), N_RULES:] += group_frac[i][:, u - 1]
    return {
        "rule_log_lik": rule_ll,
        "group_log_lik": log_lik,
        "group_log_size": log_size,
        "set_log_n": np.log(np.array([len(s) for s in unique], dtype="float64")),
        "cell_set": np.array([i for i, _ in unique_cells], dtype="int64"),
        "cell_member": member,
        "cell_valid": valid,
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
