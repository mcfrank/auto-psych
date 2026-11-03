"""Bayesian concept learning with mis-weighted evidence.

People do Bayesian concept learning over 45 rules (prior organised by family
of rule, as in `rule_family_prior_concepts`) and all intervals of 1..100,
with the size principle (each example from the concept with probability 0.9,
uniform on 1..100 otherwise). The one distortion: the likelihood of the
examples is raised to an inferred power beta (beta > 1 = over-weighting the
evidence, generalisation narrows faster as examples accumulate; beta < 1 =
conservatism). "Yes" is the posterior probability the concept contains the
target.
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


def rule_family(name):
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EXAMPLE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)
MAX_K = 4
# An interval's likelihood depends only on its size s and how many examples c
# it contains, so intervals are grouped by (s, c): 100 x (MAX_K + 1) groups.
N_GROUPS = 100 * (MAX_K + 1)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (sets, groups): log P(D | interval in group)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # log number of intervals (-1e4 if none)
    # One cell per distinct (set, target) asked about.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # target in rule
    cell_group_member = pm.Data("cell_group_member", np.zeros((1, N_GROUPS)))  # share of group's intervals containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    evidence_weight = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.5)

    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + evidence_weight * rule_log_lik
    log_groups = pt.log1p(-lambda_0) - 2 * np.log(100) + group_log_count + evidence_weight * group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_posterior = posterior[cell_set]
    in_concept = (cell_posterior * pt.concatenate([cell_rule_member, cell_group_member], axis=1)).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n = len(unique)
    rule_ll = np.stack([RULE_EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    g_ll = np.zeros((n, N_GROUPS))
    g_cnt = np.zeros((n, N_GROUPS))
    g_mem = np.zeros((n, N_GROUPS, 100))
    for i, s in enumerate(unique):
        ex = np.array(s)
        k = len(ex)
        for size in range(1, 101):
            for c in range(MAX_K + 1):
                g = (size - 1) * (MAX_K + 1) + c
                g_ll[i, g] = c * np.log(ALPHA / size + (1 - ALPHA) / 100) + (k - c) * np.log((1 - ALPHA) / 100)
        for m in range(1, 101):
            for hi in range(m, 101):
                size = hi - m + 1
                c = int(((ex >= m) & (ex <= hi)).sum())
                g = (size - 1) * (MAX_K + 1) + c
                g_cnt[i, g] += 1
                g_mem[i, g, m - 1:hi] += 1
    nz = g_cnt > 0
    g_mem[nz] /= g_cnt[nz][:, None]
    log_cnt = np.where(nz, np.log(np.maximum(g_cnt, 1)), -1e4)
    g_ll = np.where(nz, g_ll, 0.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: j for j, k in enumerate(cells)}
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells], dtype="int64")
    return {
        "rule_log_lik": rule_ll,
        "group_log_lik": g_ll,
        "group_log_count": log_cnt,
        "cell_set": cell_set,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_target].T.copy(),
        "cell_group_member": g_mem[cell_set, :, cell_target],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
