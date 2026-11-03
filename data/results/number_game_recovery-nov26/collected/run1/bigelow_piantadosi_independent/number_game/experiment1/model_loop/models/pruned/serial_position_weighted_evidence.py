"""Serial-position-weighted evidence in tempered Bayesian concept learning.

People do tempered Bayesian concept learning over 45 arithmetic rules and all
intervals of 1..100 (prior lambda_0 over rules, the rest over intervals; each
example is on-concept with prob 1 - eps, else uniform noise; size-principle
evidence tempered by beta), but each example's evidence is weighted by its
position in the order shown: w_i = n * softmax(gamma * r_i), r_i the relative
position from 0 (first) to 1 (last). gamma > 0 is recency, gamma < 0 primacy;
the total evidence is unchanged, so gamma only redistributes it by order.
"Yes" is the posterior probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EXAMPLES = 4


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
# Intervals grouped by (size s, which example positions fall inside): exact,
# since the weighted likelihood depends only on these.
N_MASKS = 2**MAX_EXAMPLES
SIZES = np.arange(1, 101)
N_GROUPS = 100 * N_MASKS
GROUP_SIZE = np.repeat(SIZES, N_MASKS).astype("float64")
MASK_BITS = np.array([[(m >> i) & 1 for i in range(MAX_EXAMPLES)] for m in range(N_MASKS)], dtype="float64")
GROUP_BITS = np.tile(MASK_BITS, (100, 1))  # (groups, positions)

with pm.Model() as model:
    rule_pos = pm.Data("rule_pos", np.zeros((1, N_RULES, MAX_EXAMPLES)))  # example i inside rule
    pos_rel = pm.Data("pos_rel", np.zeros((1, MAX_EXAMPLES)))  # relative position 0..1
    pos_valid = pm.Data("pos_valid", np.ones((1, MAX_EXAMPLES)))  # 1 for shown positions
    set_n = pm.Data("set_n", np.ones(1))
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_group_in_concept = pm.Data("cell_group_in_concept", np.zeros((1, N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)
    eps = pm.Beta("eps", 2, 5)
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.5)  # serial-position gradient

    # Position weights, summing to n within each set.
    score = gamma * pos_rel + (pos_valid - 1.0) * 50.0
    w = pt.exp(score - pt.logsumexp(score, axis=1, keepdims=True)) * set_n[:, None] * pos_valid

    log_noise = pt.log(eps) - np.log(100)
    n_col = set_n[:, None]
    rule_kw = (rule_pos * w[:, None, :]).sum(axis=2)  # (sets, rules)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    rule_log_lik = rule_kw * rule_in[None, :] + (n_col - rule_kw) * log_noise
    group_kw = pt.dot(w, GROUP_BITS.T)  # (sets, groups)
    group_in = pt.log((1 - eps) / GROUP_SIZE + eps / 100)
    group_log_lik = group_kw * group_in[None, :] + (n_col - group_kw) * log_noise

    log_rules = pt.log(lambda_0) - np.log(N_RULES) + beta * rule_log_lik
    log_groups = pt.log1p(-lambda_0) - 2 * np.log(100) + beta * group_log_lik + group_log_count
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))[cell_set]
    in_concept = (posterior[:, :N_RULES] * cell_rule_member).sum(axis=1) + (
        posterior[:, N_RULES:] * cell_group_in_concept
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _interval_groups(examples):
    count = np.zeros(N_GROUPS)
    in_concept = np.zeros((N_GROUPS, 100))
    ex = np.array(examples)
    bits = 2 ** np.arange(len(ex))
    for s in SIZES:
        for m in range(1, 102 - s):
            mask = int((((ex >= m) & (ex <= m + s - 1)) * bits).sum())
            g = (s - 1) * N_MASKS + mask
            count[g] += 1
            in_concept[g, m - 1 : m + s - 1] += 1
    nonzero = count > 0
    in_concept[nonzero] /= count[nonzero, None]
    log_count = np.where(nonzero, np.log(np.maximum(count, 1)), -1e4)
    return log_count, in_concept


def _positions(s):
    n = len(s)
    rel = np.zeros(MAX_EXAMPLES)
    valid = np.zeros(MAX_EXAMPLES)
    valid[:n] = 1.0
    rel[:n] = np.arange(n) / (n - 1) if n > 1 else 0.5
    return rel, valid


def _rule_pos(s):
    out = np.zeros((N_RULES, MAX_EXAMPLES))
    out[:, : len(s)] = RULE_MEMBERSHIP[:, np.array(s) - 1]
    return out


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    groups = [_interval_groups(s) for s in unique]
    pos = [_positions(s) for s in unique]
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "rule_pos": np.stack([_rule_pos(s) for s in unique]),
        "pos_rel": np.stack([p[0] for p in pos]),
        "pos_valid": np.stack([p[1] for p in pos]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "group_log_count": np.stack([g[0] for g in groups]),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_group_in_concept": np.stack([groups[i][1][:, t - 1] for i, t in cells]),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
