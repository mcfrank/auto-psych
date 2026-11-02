"""Tempered-posterior commitment over concepts.

People's belief over candidate concepts differs from Bayesian averaging by a
single tempering exponent: the rule-and-interval posterior is raised to a
power beta and renormalised. beta > 1 commits to the best concept (beta -> inf
is the MAP concept), beta = 1 is full Bayesian averaging, beta < 1 hedges
conservatively across weaker concepts. It disagrees most with the averaging
model where a rule and an interval compete (targets just outside the range,
numeric neighbours fitting no rule): the runner-up concepts' weight decides.

Hypothesis space, prior and likelihood are those of the rule + interval
concept learner (45 rules, all intervals of 1..100, rule prior lambda_0
split equally, each interval (1 - lambda_0) / 100^2, size principle with 10%
uniform noise).
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
INTERVAL_LOG_PRIOR_OFFSET = -2 * np.log(100)

# Intervals with the same log P(D | h) under a set have the same tempered
# posterior, so per set they are merged exactly into groups (log group size,
# and per group the fraction of its intervals containing the target).
# prepare_observed builds these once per set, padded with empty groups, keeping
# only the targets each set was asked about (slots padded with membership 0).

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    rule_in_concept = pm.Data("rule_in_concept", np.zeros((1, N_RULES, 1)))  # (sets, rules, target slots)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, 1)))  # (sets, groups)
    group_log_count = pm.Data("group_log_count", np.zeros((1, 1)))  # (sets, groups); -1e4 = padding
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, 1, 1)))  # (sets, groups, target slots)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * slots + target slot
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.75)  # tempering exponent

    log_rules = beta * (pt.log(lambda_0) - np.log(N_RULES) + rule_log_lik)
    log_groups = beta * (pt.log1p(-lambda_0) + INTERVAL_LOG_PRIOR_OFFSET + group_log_lik) + group_log_count
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, hyps)
    membership = pt.concatenate([rule_in_concept, group_in_concept], axis=1)
    in_concept = pt.batched_dot(posterior[:, None, :], membership)[:, 0, :]  # (sets, target slots)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    interval_membership = MEMBERSHIP[N_RULES:]
    rule_ll, groups = [], []
    for s in unique:
        ll = EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        rule_ll.append(ll[:N_RULES])
        keys, inverse = np.unique(np.round(ll[N_RULES:], 9), return_inverse=True)
        counts = np.bincount(inverse, minlength=len(keys)).astype("float64")
        members = np.zeros((len(keys), 100))
        np.add.at(members, inverse, interval_membership)
        groups.append((keys, np.log(counts), members / counts[:, None]))
    asked = [list(dict.fromkeys(int(r["target"]) - 1 for s2, r in zip(sets, rows) if s2 == s)) for s in unique]
    slot = [{t: k for k, t in enumerate(a)} for a in asked]
    n_slots = max(len(a) for a in asked)
    n_groups = max(len(g[0]) for g in groups)
    r_in = np.zeros((len(unique), N_RULES, n_slots))
    g_ll = np.zeros((len(unique), n_groups))
    g_lc = np.full((len(unique), n_groups), -1e4)
    g_in = np.zeros((len(unique), n_groups, n_slots))
    for i, (keys, log_counts, frac) in enumerate(groups):
        a = np.array(asked[i])
        r_in[i, :, : len(a)] = RULE_MEMBERSHIP[:, a]
        g_ll[i, : len(keys)], g_lc[i, : len(keys)] = keys, log_counts
        g_in[i, : len(keys), : len(a)] = frac[:, a]
    return {
        "rule_log_lik": np.stack(rule_ll),
        "rule_in_concept": r_in,
        "group_log_lik": g_ll,
        "group_log_count": g_lc,
        "group_in_concept": g_in,
        "row_cell": np.array(
            [index[s] * n_slots + slot[index[s]][int(r["target"]) - 1] for s, r in zip(sets, rows)], dtype="int64"
        ),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
